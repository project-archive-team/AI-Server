from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import GEMINI_CHAT_MODEL, GEMINI_EMBEDDING_MODEL
from ragas_eval import (
    METRIC_INPUTS,
    _score_value,
    build_ragas_metrics,
    collect_rag_samples,
    load_cases,
    summarize_results,
)
import services


QUOTA_MARKERS = (
    "429",
    "resource_exhausted",
    "resource exhausted",
    "quota",
    "rate limit",
    "rate_limit",
    "too many requests",
)
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
TRANSIENT_MARKERS = QUOTA_MARKERS + (
    "unavailable",
    "high demand",
    "temporarily unavailable",
    "internal server error",
)
RETRY_DELAY_PATTERNS = (
    re.compile(r"retry(?:\s+after|\s+in)?\s*[:=]?\s*([0-9.]+)\s*s", re.IGNORECASE),
    re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?([0-9.]+)s", re.IGNORECASE),
)
DAILY_QUOTA_MARKERS = (
    "requestsperday",
    "requests per day",
    "perday",
)


def is_quota_error(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "code", None) or getattr(current, "status_code", None)
        text = f"{type(current).__name__}: {current}".lower()
        if code == 429 or any(marker in text for marker in QUOTA_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def is_retryable_api_error(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "code", None) or getattr(current, "status_code", None)
        text = f"{type(current).__name__}: {current}".lower()
        if code in TRANSIENT_STATUS_CODES or any(marker in text for marker in TRANSIENT_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def quota_wait_seconds(error: BaseException, default_seconds: int) -> int:
    text = str(error)
    normalized_text = text.lower().replace("_", " ")
    if any(marker in normalized_text for marker in DAILY_QUOTA_MARKERS):
        return max(1, default_seconds)
    for pattern in RETRY_DELAY_PATTERNS:
        match = pattern.search(text)
        if match:
            return max(1, int(float(match.group(1))) + 1)
    return max(1, default_seconds)


def write_checkpoint(
    path: Path,
    dataset: Path,
    samples: list[dict[str, Any]],
    status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset": str(dataset.resolve()),
        "models": {
            "rag": GEMINI_CHAT_MODEL,
            "embedding": GEMINI_EMBEDDING_MODEL,
            "evaluator": os.getenv("RAGAS_EVALUATOR_MODEL", GEMINI_CHAT_MODEL),
        },
        "summary": summarize_results(samples),
        "samples": samples,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_checkpoint(path: Path, dataset: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    saved_dataset = Path(payload.get("dataset", ""))
    if saved_dataset.resolve() != dataset.resolve():
        raise ValueError("체크포인트의 데이터셋이 현재 데이터셋과 다릅니다.")
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("체크포인트 samples 형식이 올바르지 않습니다.")
    return samples


def wait_for_quota(error: BaseException, default_seconds: int) -> None:
    fallback_seconds = default_seconds if is_quota_error(error) else min(default_seconds, 60)
    seconds = quota_wait_seconds(error, fallback_seconds)
    retry_at = datetime.fromtimestamp(time.time() + seconds).astimezone().isoformat(timespec="seconds")
    reason = "무료 쿼터 제한" if is_quota_error(error) else "일시적 Gemini API 장애"
    print(
        f"[retry] {reason} 감지: {seconds}초 대기 후 {retry_at}에 재개합니다.",
        flush=True,
    )
    time.sleep(seconds)


def run(args: argparse.Namespace) -> int:
    cases = load_cases(args.dataset)
    if args.max_samples is not None:
        cases = cases[: args.max_samples]
    if args.project_id is not None:
        cases = [case.model_copy(update={"projectId": args.project_id}) for case in cases]
    if args.vector_store is not None:
        services.STORE_PATH = args.vector_store.resolve()
    samples = load_checkpoint(args.output, args.dataset) if args.resume else []
    if len(samples) > len(cases):
        raise ValueError("체크포인트 문항 수가 현재 데이터셋보다 많습니다.")

    metrics = None
    print(
        f"[start] 총 {len(cases)}문항, 기존 수집 {len(samples)}문항, 출력 {args.output}",
        flush=True,
    )

    while len(samples) < len(cases):
        index = len(samples)
        try:
            sample = collect_rag_samples(
                [cases[index]],
                default_user_id=args.user_id,
            )[0]
            sample["caseIndex"] = index
            sample["scores"] = {}
            samples.append(sample)
            write_checkpoint(args.output, args.dataset, samples, "running")
            print(f"[collect] {index + 1}/{len(cases)} 완료", flush=True)
            if args.request_delay_seconds:
                time.sleep(args.request_delay_seconds)
        except Exception as error:
            if not is_retryable_api_error(error):
                raise
            write_checkpoint(args.output, args.dataset, samples, "waiting_for_quota")
            wait_for_quota(error, args.quota_wait_seconds)

    if args.collect_only:
        write_checkpoint(args.output, args.dataset, samples, "complete")
        return 0

    metrics = build_ragas_metrics()
    for sample_index, sample in enumerate(samples):
        scores = sample.setdefault("scores", {})
        errors = sample.setdefault("errors", {})
        for metric_name, metric in metrics.items():
            if scores.get(metric_name) is not None:
                continue
            metric_input = {
                field: sample[field]
                for field in METRIC_INPUTS[metric_name]
            }
            while True:
                try:
                    value = _score_value(metric.score(**metric_input))
                    if value is None:
                        raise RuntimeError("RAGAS가 유효한 숫자 점수를 반환하지 않았습니다.")
                    scores[metric_name] = value
                    errors.pop(metric_name, None)
                    if not errors:
                        sample.pop("errors", None)
                    write_checkpoint(args.output, args.dataset, samples, "running")
                    print(
                        f"[judge] {sample_index + 1}/{len(samples)} {metric_name}={value:.4f}",
                        flush=True,
                    )
                    if args.request_delay_seconds:
                        time.sleep(args.request_delay_seconds)
                    break
                except Exception as error:
                    if is_retryable_api_error(error):
                        write_checkpoint(args.output, args.dataset, samples, "waiting_for_quota")
                        wait_for_quota(error, args.quota_wait_seconds)
                        continue
                    errors[metric_name] = str(error)
                    write_checkpoint(args.output, args.dataset, samples, "running_with_errors")
                    print(f"[error] {sample_index + 1} {metric_name}: {error}", flush=True)
                    break

    has_errors = any(sample.get("errors") for sample in samples)
    write_checkpoint(
        args.output,
        args.dataset,
        samples,
        "complete_with_errors" if has_errors else "complete",
    )
    return 1 if has_errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="무료 Gemini 쿼터에 맞춰 체크포인트부터 자동 재개하는 RAGAS 러너"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-id", type=int, default=0)
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="데이터셋의 projectId를 실행 시 지정한 값으로 재매핑합니다.",
    )
    parser.add_argument(
        "--vector-store",
        type=Path,
        default=None,
        help="운영 데이터와 분리된 평가용 vector_store.json 경로",
    )
    parser.add_argument("--quota-wait-seconds", type=int, default=3600)
    parser.add_argument("--request-delay-seconds", type=float, default=3.0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="데이터셋 앞에서부터 평가할 최대 문항 수",
    )
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--collect-only", action="store_true")
    parser.set_defaults(resume=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[stop] 중단되었습니다. 다음 실행 시 체크포인트부터 재개합니다.")
        return 130
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[fatal] {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
