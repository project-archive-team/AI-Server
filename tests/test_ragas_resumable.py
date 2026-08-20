import json
from pathlib import Path
import sys

from google.genai import errors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ragas_resumable import (
    build_parser,
    is_quota_error,
    is_retryable_api_error,
    load_checkpoint,
    quota_wait_seconds,
    write_checkpoint,
)


def test_quota_error_detection_and_retry_delay() -> None:
    error = errors.ClientError(
        429,
        {"error": {"message": "Quota exceeded. Please retry in 42.5s"}},
    )

    assert is_quota_error(error)
    assert quota_wait_seconds(error, 3600) == 43
    assert not is_quota_error(ValueError("bad dataset"))
    assert is_retryable_api_error(
        errors.ServerError(503, {"error": {"message": "high demand"}})
    )
    assert not is_retryable_api_error(ValueError("bad dataset"))

    daily_error = errors.ClientError(
        429,
        {"error": {"message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}},
    )
    assert quota_wait_seconds(daily_error, 3600) == 3600


def test_checkpoint_round_trip(tmp_path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")
    output = tmp_path / "checkpoint.json"
    samples = [
        {
            "responseTimeMs": 100.0,
            "scores": {"faithfulness": 0.8},
        }
    ]

    write_checkpoint(output, dataset, samples, "running")

    assert load_checkpoint(output, dataset) == samples
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "running"
    assert payload["summary"]["sampleCount"] == 1


def test_parser_accepts_project_and_store_override(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "dataset.json",
            "--output",
            "result.json",
            "--project-id",
            "14",
            "--vector-store",
            str(tmp_path / "vector_store.json"),
            "--max-samples",
            "40",
        ]
    )

    assert args.project_id == 14
    assert args.vector_store == tmp_path / "vector_store.json"
    assert args.max_samples == 40
