from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_ragas_core_checkpoint import extract_primary_response, prepare_samples


def test_extracts_interview_core_without_follow_up() -> None:
    response = """## 핵심 답변
근거 기반 핵심 답변입니다.

## 30초 답변 예시
짧은 답변입니다.

## 예상 꼬리 질문과 추천 답변
추가 내용입니다."""

    assert extract_primary_response(response, "interview") == "근거 기반 핵심 답변입니다."


def test_extracts_core_with_deeper_markdown_heading() -> None:
    response = """### 핵심 답변
근거 기반 핵심 답변입니다.

### 프로젝트에서 확인된 내용
후속 내용입니다.

## 참고 자료
- README.md"""

    assert extract_primary_response(response, "general") == "근거 기반 핵심 답변입니다."


def test_extracts_portfolio_analysis() -> None:
    response = """## 프로젝트 분석
프로젝트 핵심 분석입니다.

## 포트폴리오용 문장
포트폴리오 문장입니다."""

    assert extract_primary_response(response, "portfolio") == "프로젝트 핵심 분석입니다."


def test_general_response_falls_back_to_preface() -> None:
    response = """질문에 직접 답하는 첫 본문입니다.

## 프로젝트에서 확인된 내용
추가 설명입니다."""

    assert extract_primary_response(response, "general") == "질문에 직접 답하는 첫 본문입니다."


def test_prepared_samples_clear_previous_evaluation() -> None:
    samples = [
        {
            "answerMode": "interview",
            "response": "## 핵심 답변\n핵심\n\n## 30초 답변 예시\n요약",
            "scores": {"faithfulness": 0.2},
            "errors": {"context_recall": "error"},
        }
    ]

    prepared = prepare_samples(samples)

    assert prepared[0]["response"] == "핵심"
    assert prepared[0]["scores"] == {}
    assert prepared[0]["evaluationScope"] == "primary_response_only"
    assert "errors" not in prepared[0]
