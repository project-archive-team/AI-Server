import json

import pytest

from ragas_eval import (
    EvaluationCase,
    collect_rag_samples,
    failed_thresholds,
    load_cases,
    parse_thresholds,
    run_ragas_evaluation,
    summarize_results,
)


def test_load_cases_validates_dataset(tmp_path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "projectId": 7,
                    "question": "사용 기술은 무엇인가요?",
                    "reference": "FastAPI를 사용했습니다.",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_cases(dataset)

    assert cases[0].projectId == 7
    assert cases[0].answerMode == "general"
    assert cases[0].topK == 8


def test_load_cases_rejects_empty_dataset(tmp_path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="한 개 이상의 항목"):
        load_cases(dataset)


def test_collect_rag_samples_uses_real_contract_shape() -> None:
    captured = {}
    times = iter([10.0, 11.234])

    def retriever(request):
        captured["request"] = request
        return [{"text": "FastAPI를 사용했습니다."}]

    def generator(request, documents):
        assert documents[0]["text"] == "FastAPI를 사용했습니다."
        return "AI 서버는 FastAPI로 구현했습니다."

    samples = collect_rag_samples(
        [
            EvaluationCase(
                projectId=7,
                question="AI 서버 기술은 무엇인가요?",
                reference="FastAPI를 사용했습니다.",
                topK=3,
            )
        ],
        retriever=retriever,
        answer_generator=generator,
        clock=lambda: next(times),
    )

    assert captured["request"].project_id == 7
    assert captured["request"].top_k == 3
    assert samples[0]["retrieved_contexts"] == ["FastAPI를 사용했습니다."]
    assert samples[0]["response"] == "AI 서버는 FastAPI로 구현했습니다."
    assert samples[0]["responseTimeMs"] == 1234.0


class _Score:
    def __init__(self, value):
        self.value = value


class _Metric:
    def __init__(self, value):
        self.value = value
        self.inputs = None

    def score(self, **kwargs):
        self.inputs = kwargs
        return _Score(self.value)


def test_run_ragas_evaluation_routes_metric_inputs() -> None:
    faithfulness = _Metric(0.8)
    context_recall = _Metric(0.6)
    sample = {
        "projectId": 7,
        "answerMode": "general",
        "user_input": "질문",
        "retrieved_contexts": ["근거"],
        "response": "답변",
        "reference": "기준 답변",
    }

    evaluated = run_ragas_evaluation(
        [sample],
        metrics={
            "faithfulness": faithfulness,
            "context_recall": context_recall,
        },
    )

    assert faithfulness.inputs == {
        "user_input": "질문",
        "response": "답변",
        "retrieved_contexts": ["근거"],
    }
    assert context_recall.inputs == {
        "user_input": "질문",
        "reference": "기준 답변",
        "retrieved_contexts": ["근거"],
    }
    assert evaluated[0]["scores"] == {
        "faithfulness": 0.8,
        "context_recall": 0.6,
    }


def test_summary_and_threshold_failure() -> None:
    summary = summarize_results(
        [
            {"scores": {"faithfulness": 0.8, "context_recall": 0.6}, "responseTimeMs": 1000.0},
            {"scores": {"faithfulness": 1.0, "context_recall": 0.8}, "responseTimeMs": 1500.0},
        ]
    )
    thresholds = parse_thresholds(
        ["faithfulness=0.85", "context_recall=0.75"]
    )

    assert summary["metrics"] == {
        "context_recall": 0.7,
        "faithfulness": 0.9,
    }
    assert summary["averageResponseTimeMs"] == 1250.0
    assert failed_thresholds(summary, thresholds) == [
        "context_recall: 0.7 < 0.75"
    ]
