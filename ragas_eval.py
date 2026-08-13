from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from config import (
    GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    client,
)
from schemas import ChatRequest
from services import generate_answer, retrieve_project_context


METRIC_INPUTS = {
    "faithfulness": ("user_input", "response", "retrieved_contexts"),
    "answer_relevancy": ("user_input", "response"),
    "context_precision": ("user_input", "reference", "retrieved_contexts"),
    "context_recall": ("user_input", "reference", "retrieved_contexts"),
}


class EvaluationCase(BaseModel):
    projectId: int
    question: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    answerMode: Literal["general", "portfolio", "interview"] = "general"
    topK: int = Field(default=8, ge=1, le=10)


def load_cases(path: Path) -> list[EvaluationCase]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"평가 데이터셋을 찾을 수 없습니다: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"평가 데이터셋 JSON이 올바르지 않습니다: {error}") from error

    if not isinstance(raw, list) or not raw:
        raise ValueError("평가 데이터셋은 한 개 이상의 항목을 가진 JSON 배열이어야 합니다.")

    try:
        return [EvaluationCase.model_validate(item) for item in raw]
    except ValidationError as error:
        raise ValueError(f"평가 데이터셋 항목이 계약과 맞지 않습니다:\n{error}") from error


def collect_rag_samples(
    cases: list[EvaluationCase],
    retriever: Optional[Callable[[ChatRequest], list[dict[str, Any]]]] = None,
    answer_generator: Optional[
        Callable[[ChatRequest, list[dict[str, Any]]], str]
    ] = None,
    clock: Callable[[], float] = perf_counter,
) -> list[dict[str, Any]]:
    retrieve = retriever or retrieve_project_context
    generate = answer_generator or generate_answer
    samples: list[dict[str, Any]] = []

    for case in cases:
        request = ChatRequest(
            user_id=0,
            project_id=case.projectId,
            question=case.question,
            answer_mode=case.answerMode,
            top_k=case.topK,
        )
        started_at = clock()
        documents = retrieve(request)
        response = generate(request, documents)
        response_time_ms = round((clock() - started_at) * 1000, 2)
        samples.append(
            {
                "projectId": case.projectId,
                "answerMode": case.answerMode,
                "user_input": case.question,
                "retrieved_contexts": [
                    document.get("text", "") for document in documents
                ],
                "response": response,
                "responseTimeMs": response_time_ms,
                "reference": case.reference,
            }
        )

    return samples


def build_ragas_metrics() -> dict[str, Any]:
    try:
        from ragas.embeddings import GoogleEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as error:
        raise RuntimeError(
            "RAGAS 평가 의존성이 없습니다. "
            "pip install -r requirements-eval.txt 를 먼저 실행해 주세요."
        ) from error

    evaluator_model = os.getenv("RAGAS_EVALUATOR_MODEL", GEMINI_CHAT_MODEL)
    evaluator_llm = llm_factory(
        evaluator_model,
        provider="google",
        client=client,
    )
    evaluator_embeddings = GoogleEmbeddings(
        client=client,
        model=GEMINI_EMBEDDING_MODEL,
    )
    return {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }


def _score_value(result: Any) -> Optional[float]:
    value = getattr(result, "value", result)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def run_ragas_evaluation(
    samples: list[dict[str, Any]],
    metrics: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    scorers = metrics or build_ragas_metrics()
    evaluated: list[dict[str, Any]] = []

    for sample in samples:
        scores: dict[str, Optional[float]] = {}
        errors: dict[str, str] = {}
        for metric_name, metric in scorers.items():
            required_fields = METRIC_INPUTS.get(metric_name)
            if required_fields is None:
                raise ValueError(f"지원하지 않는 RAGAS 지표입니다: {metric_name}")
            metric_input = {field: sample[field] for field in required_fields}
            try:
                scores[metric_name] = _score_value(metric.score(**metric_input))
                if scores[metric_name] is None:
                    errors[metric_name] = "RAGAS가 유효한 숫자 점수를 반환하지 않았습니다."
            except Exception as error:
                scores[metric_name] = None
                errors[metric_name] = str(error)

        evaluated_sample = {**sample, "scores": scores}
        if errors:
            evaluated_sample["errors"] = errors
        evaluated.append(evaluated_sample)

    return evaluated


def summarize_results(samples: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            metric_name
            for sample in samples
            for metric_name in sample.get("scores", {})
        }
    )
    metric_averages: dict[str, Optional[float]] = {}
    for metric_name in metric_names:
        values = [
            sample["scores"].get(metric_name)
            for sample in samples
            if sample.get("scores", {}).get(metric_name) is not None
        ]
        metric_averages[metric_name] = round(mean(values), 4) if values else None

    return {
        "sampleCount": len(samples),
        "successfulSampleCount": sum(1 for sample in samples if not sample.get("errors")),
        "metrics": metric_averages,
        "averageResponseTimeMs": round(
            mean(sample["responseTimeMs"] for sample in samples), 2
        ) if samples else None,
    }


def parse_thresholds(values: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values:
        try:
            metric_name, raw_score = value.split("=", 1)
            score = float(raw_score)
        except ValueError as error:
            raise ValueError(
                f"임계값은 metric=score 형식이어야 합니다: {value}"
            ) from error
        if metric_name not in METRIC_INPUTS:
            raise ValueError(f"지원하지 않는 RAGAS 지표입니다: {metric_name}")
        if not 0 <= score <= 1:
            raise ValueError(f"임계값은 0과 1 사이여야 합니다: {value}")
        thresholds[metric_name] = score
    return thresholds


def failed_thresholds(
    summary: dict[str, Any],
    thresholds: dict[str, float],
) -> list[str]:
    failures = []
    scores = summary.get("metrics", {})
    for metric_name, minimum in thresholds.items():
        actual = scores.get(metric_name)
        if actual is None or actual < minimum:
            failures.append(
                f"{metric_name}: {actual if actual is not None else 'N/A'} < {minimum}"
            )
    return failures


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / "ragas_results" / f"ragas-{stamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="실제 프로젝트 RAG 파이프라인을 RAGAS로 배치 평가합니다."
    )
    parser.add_argument("dataset", type=Path, help="골든 평가 데이터셋 JSON 경로")
    parser.add_argument("--output", type=Path, default=None, help="평가 결과 JSON 경로")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="METRIC=SCORE",
        help="평균 점수가 기준 미만이면 종료 코드 1을 반환합니다. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="RAG 응답과 검색 문맥만 수집하고 RAGAS judge 평가는 생략합니다.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        thresholds = parse_thresholds(args.threshold)
        cases = load_cases(args.dataset)
        collected = collect_rag_samples(cases)
        samples = collected if args.collect_only else run_ragas_evaluation(collected)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    summary = summarize_results(samples)
    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "models": {
            "rag": GEMINI_CHAT_MODEL,
            "embedding": GEMINI_EMBEDDING_MODEL,
            "evaluator": os.getenv("RAGAS_EVALUATOR_MODEL", GEMINI_CHAT_MODEL),
        },
        "summary": summary,
        "samples": samples,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), **summary}, ensure_ascii=False, indent=2))

    failures = failed_thresholds(summary, thresholds)
    if failures:
        print("RAGAS 품질 기준 미달: " + ", ".join(failures))
        return 1
    if any(sample.get("errors") for sample in samples):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
