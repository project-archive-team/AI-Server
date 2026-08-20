from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ragas_resumable import write_checkpoint


PRIMARY_HEADINGS = {
    "interview": ("핵심 답변", "30초 답변 예시"),
    "portfolio": ("프로젝트 분석",),
    "general": ("핵심 답변",),
}


def extract_section(response: str, heading: str) -> str | None:
    pattern = re.compile(
        rf"(?ms)^#{{2,6}}\s+{re.escape(heading)}\s*$\n(.*?)(?=^#{{2,6}}\s+|\Z)"
    )
    match = pattern.search(response)
    return match.group(1).strip() if match else None


def extract_primary_response(response: str, answer_mode: str) -> str:
    for heading in PRIMARY_HEADINGS.get(answer_mode, ("핵심 답변",)):
        section = extract_section(response, heading)
        if section:
            return section

    preface = re.split(r"(?m)^#{2,6}\s+", response, maxsplit=1)[0].strip()
    if preface:
        return preface
    raise ValueError(f"{answer_mode} 응답에서 평가할 핵심 본문을 찾지 못했습니다.")


def prepare_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for sample in samples:
        transformed = {
            **sample,
            "response": extract_primary_response(
                sample["response"], sample.get("answerMode", "general")
            ),
            "scores": {},
            "evaluationScope": "primary_response_only",
        }
        transformed.pop("errors", None)
        prepared.append(transformed)
    return prepared


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="수집된 RAG 응답의 핵심 본문만 추출해 별도 RAGAS 체크포인트를 만듭니다."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists():
        raise SystemExit(f"출력 파일이 이미 존재합니다: {args.output}")

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    dataset = payload.get("dataset")
    if not isinstance(samples, list) or not samples:
        raise SystemExit("원본 체크포인트에 평가할 samples가 없습니다.")
    if not dataset:
        raise SystemExit("원본 체크포인트에 dataset 경로가 없습니다.")

    prepared = prepare_samples(samples)
    write_checkpoint(args.output, Path(dataset), prepared, "prepared")
    print(f"[prepared] 핵심 본문 {len(prepared)}건 -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
