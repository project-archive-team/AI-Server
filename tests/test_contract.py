from datetime import datetime, timezone

import pytest
from google.genai import errors
from fastapi.testclient import TestClient
from pydantic import ValidationError

import ai_contract
import services
from ai_app import app
from services import SimpleVectorStore


def test_portfolio_contribution_requires_a_visible_achievement() -> None:
    with pytest.raises(ValidationError):
        services.PortfolioContributionSchema(
            title="통합 인증 및 권한 관리 시스템 구축",
            description="JWT와 OAuth2 기반 인증을 구현했습니다.",
            metrics=[],
        )

    contribution = services.PortfolioContributionSchema(
        title="통합 인증 및 권한 관리 시스템 구축",
        description="JWT와 OAuth2 기반 인증을 구현했습니다.",
        metrics=["여러 인증 제공자를 하나의 로그인 흐름으로 통합"],
    )

    assert contribution.metrics == ["여러 인증 제공자를 하나의 로그인 흐름으로 통합"]


def test_gemini_generation_retries_temporary_server_error(monkeypatch) -> None:
    attempts = []
    delays = []
    expected = object()

    def generate_content(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise errors.ServerError(503, {"error": {"message": "high demand"}})
        return expected

    monkeypatch.setattr(services.client.models, "generate_content", generate_content)

    result = services.generate_content_with_retry(
        model="test-model",
        contents="질문",
        config=services.types.GenerateContentConfig(),
        sleep=delays.append,
    )

    assert result is expected
    assert len(attempts) == 2
    assert delays == [2.0]


def test_gemini_generation_does_not_retry_non_transient_error(monkeypatch) -> None:
    attempts = []

    def generate_content(**kwargs):
        attempts.append(kwargs)
        raise errors.ClientError(400, {"error": {"message": "bad request"}})

    monkeypatch.setattr(services.client.models, "generate_content", generate_content)

    with pytest.raises(errors.ClientError):
        services.generate_content_with_retry(
            model="test-model",
            contents="질문",
            config=services.types.GenerateContentConfig(),
            sleep=lambda _: None,
        )

    assert len(attempts) == 1


def test_portfolio_contribution_selection_rule_requires_significant_verified_work() -> None:
    rule = services.PORTFOLIO_CONTRIBUTION_SELECTION_RULE

    assert "지원자가 직접 책임지고 수행한 사실" in rule
    assert "지원 직무 연관성, 본인 기여도, 프로젝트 영향도" in rule
    assert "단순 참여, 반복 업무, 보조 작업, 사소한 수정" in rule
    assert "개수를 억지로 채우지 않습니다" in rule
    assert "최대 5개" in rule
    assert "title에는 핵심 기여" in rule


def _chunk(artifact_id: int, text: str, seq: int = 0) -> dict:
    return {
        "artifactId": artifact_id,
        "type": "DOC",
        "title": f"artifact-{artifact_id}",
        "seq": seq,
        "text": text,
    }


def test_index_replaces_existing_artifact_chunks_and_deletes_them(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(services, "STORE_PATH", tmp_path / "store.json")
    monkeypatch.setattr(
        ai_contract,
        "create_embeddings",
        lambda texts: [[float(index + 1), 0.0] for index, _ in enumerate(texts)],
    )
    client = TestClient(app)

    first = client.post(
        "/index",
        json={
            "projectId": 7,
            "projectName": "프로젝트 아카이브",
            "chunks": [_chunk(10, "old-a"), _chunk(10, "old-b", 1)],
        },
    )
    assert first.status_code == 200
    assert first.json()["indexed"] == 2

    second = client.post(
        "/index",
        json={"projectId": 7, "chunks": [_chunk(10, "new")]},
    )
    assert second.status_code == 200

    documents = SimpleVectorStore().documents
    assert [document["text"] for document in documents] == ["new"]
    assert documents[0]["metadata"]["project_name"] == "프로젝트 아카이브"

    deleted = client.post(
        "/index/delete",
        json={"projectId": 7, "artifactIds": [10]},
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": 1}
    assert SimpleVectorStore().documents == []


def test_delete_project_only_removes_that_projects_documents(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(services, "STORE_PATH", tmp_path / "store.json")
    store = SimpleVectorStore()
    store.add_documents(
        [
            {"text": "one", "embedding": [1.0], "metadata": {"project_id": 1}},
            {"text": "two", "embedding": [1.0], "metadata": {"project_id": 2}},
        ]
    )

    response = TestClient(app).delete("/index/projects/1")

    assert response.status_code == 200
    assert response.json() == {"deleted": 1}
    assert [document["metadata"]["project_id"] for document in SimpleVectorStore().documents] == [2]


def test_search_filters_documents_before_summary_cutoff(tmp_path) -> None:
    store = SimpleVectorStore(tmp_path / "store.json")
    store.add_documents(
        [
            {
                "text": "old",
                "embedding": [1.0, 0.0],
                "metadata": {
                    "user_id": 0,
                    "project_id": 3,
                    "occurred_at": "2026-01-01T00:00:00+00:00",
                },
            },
            {
                "text": "recent",
                "embedding": [1.0, 0.0],
                "metadata": {
                    "user_id": 0,
                    "project_id": 3,
                    "occurred_at": "2026-07-29T00:00:00+00:00",
                },
            },
        ]
    )

    found = store.search(
        [1.0, 0.0],
        user_id=0,
        project_id=3,
        occurred_since=datetime(2026, 7, 20, tzinfo=timezone.utc),
    )

    assert [document["text"] for document in found] == ["recent"]


def test_search_can_limit_summary_sources_to_commits_and_meetings(tmp_path) -> None:
    store = SimpleVectorStore(tmp_path / "store.json")
    store.add_documents(
        [
            {
                "text": source_type,
                "embedding": [1.0, 0.0],
                "metadata": {
                    "user_id": 0,
                    "project_id": 3,
                    "source_type": source_type,
                },
            }
            for source_type in ("COMMIT", "MEETING", "CODE", "DOC")
        ]
    )

    found = store.search(
        [1.0, 0.0],
        user_id=0,
        project_id=3,
        source_types={"COMMIT", "MEETING"},
    )

    assert {document["text"] for document in found} == {"COMMIT", "MEETING"}


def test_summary_project_display_name_uses_real_name_or_generic_phrase() -> None:
    named_document = {"metadata": {"project_name": "프로젝트 아카이브"}}
    numbered_document = {"metadata": {"project_name": "Project 14"}}
    korean_numbered_document = {"metadata": {"project_name": "프로젝트 14"}}

    assert services.resolve_project_display_name(None, [named_document]) == "프로젝트 아카이브"
    assert services.resolve_project_display_name(None, [numbered_document]) == "이 프로젝트"
    assert services.resolve_project_display_name(None, [korean_numbered_document]) == "이 프로젝트"
    assert services.resolve_project_display_name("팀 아카이브", [numbered_document]) == "팀 아카이브"


def test_summary_passes_name_and_summary_mode(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        ai_contract,
        "retrieve_project_context",
        lambda request, occurred_since=None, source_types=None: captured.update(
            {
                "request": request,
                "occurred_since": occurred_since,
                "source_types": source_types,
            }
        ) or [],
    )
    monkeypatch.setattr(ai_contract, "generate_answer", lambda request, documents: "요약")

    response = TestClient(app).post(
        "/summary",
        json={
            "projectId": 14,
            "projectName": "프로젝트 아카이브",
            "since": "2026-08-01T00:00:00Z",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"summary": "요약"}
    assert captured["request"].answer_mode == "summary"
    assert captured["request"].project_name == "프로젝트 아카이브"
    assert captured["source_types"] == {"COMMIT", "MEETING"}
    assert captured["request"].question.startswith("2026-08-01 이후")
    assert "T00:00:00" not in captured["request"].question
    assert "시·분·초와 시간대 정보는 출력하지 않습니다" in services.get_mode_instruction("summary")


def test_chat_passes_selected_project_name(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        ai_contract,
        "retrieve_project_context",
        lambda request, occurred_since=None, source_types=None: captured.update(
            {"request": request}
        ) or [],
    )
    monkeypatch.setattr(ai_contract, "generate_answer", lambda request, documents: "답변")

    response = TestClient(app).post(
        "/chat",
        json={
            "projectId": 14,
            "projectName": "학술제 프로젝트 아카이빙",
            "question": "임베딩 오류를 어떻게 해결했나요?",
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "답변"
    assert captured["request"].project_name == "학술제 프로젝트 아카이빙"


def test_generate_answer_replaces_numbered_project_metadata(monkeypatch) -> None:
    captured = {}

    class Response:
        text = "생성 답변"

    monkeypatch.setattr(
        services,
        "generate_content_with_retry",
        lambda **kwargs: captured.update(kwargs) or Response(),
    )
    request = services.ChatRequest(
        user_id=0,
        project_id=14,
        project_name="학술제 프로젝트 아카이빙",
        question="질문",
        answer_mode="general",
    )

    answer = services.generate_answer(
        request,
        [
            {
                "text": "근거",
                "score": 0.9,
                "metadata": {
                    "project_name": "Project 14",
                    "source_name": "README.md",
                    "source_type": "DOC",
                },
            }
        ],
    )

    assert answer == "생성 답변"
    assert "표시할 프로젝트명: 학술제 프로젝트 아카이빙" in captured["contents"]
    assert "프로젝트명: 학술제 프로젝트 아카이빙" in captured["contents"]
    assert "Project 14" not in captured["contents"]


def test_generate_answer_uses_generic_name_when_only_numbered_name_exists(monkeypatch) -> None:
    captured = {}

    class Response:
        text = "생성 답변"

    monkeypatch.setattr(
        services,
        "generate_content_with_retry",
        lambda **kwargs: captured.update(kwargs) or Response(),
    )
    request = services.ChatRequest(
        user_id=0,
        project_id=14,
        question="질문",
        answer_mode="general",
    )

    services.generate_answer(
        request,
        [
            {
                "text": "근거",
                "score": 0.9,
                "metadata": {
                    "project_name": "Project 14",
                    "source_name": "README.md",
                    "source_type": "DOC",
                },
            }
        ],
    )

    assert "표시할 프로젝트명: 이 프로젝트" in captured["contents"]
    assert "프로젝트명: 이 프로젝트" in captured["contents"]
    assert "Project 14" not in captured["contents"]


def test_citations_are_deduplicated_by_artifact_without_losing_metadata() -> None:
    documents = [
        {
            "text": "first chunk",
            "metadata": {
                "artifact_id": 11,
                "source_name": "README.md",
                "source_url": "https://example.com/readme",
            },
        },
        {
            "text": "second chunk",
            "metadata": {
                "artifact_id": 11,
                "source_name": "README.md",
                "source_url": "https://example.com/readme",
            },
        },
        {
            "text": "another artifact",
            "metadata": {
                "artifact_id": 12,
                "source_name": "회의록",
                "source_url": None,
            },
        },
    ]

    citations = ai_contract._citations(documents)

    assert citations == [
        {
            "artifactId": 11,
            "title": "README.md",
            "url": "https://example.com/readme",
            "snippet": "first chunk",
        },
        {
            "artifactId": 12,
            "title": "회의록",
            "url": None,
            "snippet": "another artifact",
        },
    ]


def test_interview_prompt_requests_answers_for_each_follow_up_question() -> None:
    instruction = services.get_mode_instruction("interview")

    assert "예상 꼬리 질문과 추천 답변" in instruction
    assert "정확히 3개" in instruction
    assert "각 질문 바로 아래" in instruction
    assert "**추천 답변**" in instruction
    assert "없는 사실이나 수치를 만들지 않습니다" in instruction
    assert "질문에서 직접 요구하거나" in instruction
    assert "개별 자료 삭제 API" in instruction
    assert "프로젝트 단위 삭제 API" in instruction


def test_general_prompt_stays_with_verified_question_evidence() -> None:
    instruction = services.get_mode_instruction("general")

    assert "질문에 필요한 사실만 2~5문장" in instruction
    assert "추론이나 일반적인 개선 제안으로 채우지 않습니다" in instruction
    assert "## 강점" not in instruction


def test_career_role_alignment_rule_prioritizes_verified_personal_contribution() -> None:
    rule = services.CAREER_ROLE_ALIGNMENT_RULE

    assert "지원 직무" in rule
    assert "AI, 백엔드, 프론트엔드" in rule
    assert "본인이 맡은 책임, 판단, 행동, 기술적 기여" in rule
    assert "전이 가능한 역량" in rule
    assert "수행하지 않은 역할이나 기여" in rule


def _retrieved_document() -> dict:
    return {
        "text": "Redis 캐시를 도입했고 기본 TTL을 30분으로 설정했다.",
        "score": 0.91,
        "metadata": {
            "artifact_id": 21,
            "project_id": 7,
            "source_name": "CacheService.java",
            "source_url": "https://example.com/cache",
        },
    }


def test_career_star_returns_renderable_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        ai_contract,
        "retrieve_project_context",
        lambda request: [_retrieved_document()],
    )
    monkeypatch.setattr(
        ai_contract,
        "generate_career_star",
        lambda job_role, question, documents: {
            "star": {
                "situation": "검색 요청이 반복되는 상황이었습니다.",
                "task": "반복 조회 비용을 줄여야 했습니다.",
                "action": "Redis 캐시와 30분 TTL을 적용했습니다.",
                "result": "반복 조회를 캐시에서 처리하도록 개선했습니다.",
            },
            "finalAnswer": "Redis 캐시와 30분 TTL을 적용했습니다.",
            "missingEvidence": ["응답 시간의 전후 측정 수치는 확인되지 않습니다."],
        },
    )

    response = TestClient(app).post(
        "/career/star",
        json={
            "projectId": 7,
            "jobRole": "백엔드 / AI 엔지니어",
            "question": "기술적 어려움을 해결한 경험을 설명해 주세요.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["jobRole"] == "백엔드 / AI 엔지니어"
    assert body["star"]["action"] == "Redis 캐시와 30분 TTL을 적용했습니다."
    assert body["missingEvidence"] == ["응답 시간의 전후 측정 수치는 확인되지 않습니다."]
    assert body["citations"] == [
        {
            "artifactId": 21,
            "title": "CacheService.java",
            "url": "https://example.com/cache",
            "snippet": "Redis 캐시를 도입했고 기본 TTL을 30분으로 설정했다.",
        }
    ]


def test_career_interview_questions_returns_cards_with_citations(monkeypatch) -> None:
    monkeypatch.setattr(
        ai_contract,
        "retrieve_project_context",
        lambda request: [_retrieved_document()],
    )
    monkeypatch.setattr(
        ai_contract,
        "generate_career_interview_questions",
        lambda job_role, question_count, documents: [
            {
                "category": "성능",
                "likelihood": "HIGH",
                "question": "Redis 캐시의 TTL을 30분으로 정한 이유는 무엇인가요?",
                "modelAnswer": "반복 조회를 줄이기 위해 Redis 캐시를 적용했습니다.",
                "checkpoints": ["캐시 도입 목적", "TTL 정책 이해"],
                "followUps": [
                    {
                        "question": "캐시 무효화는 어떻게 처리했나요?",
                        "recommendedAnswer": "무효화 방식은 자료에서 확인되지 않아 추가 확인이 필요합니다.",
                    },
                    {
                        "question": "도입 전후 성능 수치는 무엇인가요?",
                        "recommendedAnswer": "측정 수치는 자료에서 확인되지 않습니다.",
                    },
                ],
            }
        ],
    )

    response = TestClient(app).post(
        "/career/interview-questions",
        json={
            "projectId": 7,
            "jobRole": "백엔드 / AI 엔지니어",
            "questionCount": 1,
        },
    )

    assert response.status_code == 200
    question = response.json()["questions"][0]
    assert question["category"] == "성능"
    assert question["likelihood"] == "HIGH"
    assert len(question["followUps"]) == 2
    assert question["citations"][0]["artifactId"] == 21


def test_career_question_count_is_limited_to_five() -> None:
    response = TestClient(app).post(
        "/career/interview-questions",
        json={
            "projectId": 7,
            "jobRole": "백엔드 개발자",
            "questionCount": 6,
        },
    )

    assert response.status_code == 422


def test_career_generators_do_not_call_gemini_without_documents() -> None:
    star = services.generate_career_star("백엔드 개발자", "문항", [])
    questions = services.generate_career_interview_questions("백엔드 개발자", 3, [])

    assert star["missingEvidence"] == ["프로젝트 자료를 먼저 등록해 주세요."]
    assert questions == []


def test_portfolio_context_uses_multiple_queries_and_deduplicates(monkeypatch) -> None:
    questions = []

    def fake_retrieve(request):
        questions.append(request.question)
        return [
            {
                "id": "same-document",
                "text": "shared",
                "metadata": {"project_id": request.project_id},
            },
            {
                "id": f"document-{len(questions)}",
                "text": request.question,
                "metadata": {"project_id": request.project_id},
            },
        ]

    monkeypatch.setattr(services, "retrieve_project_context", fake_retrieve)

    documents = services.retrieve_portfolio_context(project_id=7)

    assert len(questions) == 4
    assert len(documents) == 5
    assert [document["id"] for document in documents].count("same-document") == 1


def test_portfolio_report_returns_screen_sections_and_citations(monkeypatch) -> None:
    document = _retrieved_document()
    document["metadata"]["project_name"] = "프로젝트 아카이브"
    monkeypatch.setattr(
        ai_contract,
        "retrieve_portfolio_context",
        lambda project_id: [document],
    )
    monkeypatch.setattr(
        ai_contract,
        "generate_portfolio_report",
        lambda **kwargs: {
            "oneLineSummary": "개발 산출물을 검색하고 활용하는 아카이빙 플랫폼",
            "executiveSummary": {
                "servicePurpose": "개발 산출물 통합 검색",
                "targetUsers": "개발자",
                "period": kwargs["period"],
                "teamSize": kwargs["team_size"],
                "role": kwargs["role"],
            },
            "techStack": [
                {
                    "name": "Redis",
                    "category": "ARCHITECTURE",
                    "reason": "반복 조회를 캐시하기 위해 사용",
                }
            ],
            "systemArchitecture": "애플리케이션에서 Redis 캐시를 사용합니다.",
            "dataPipeline": "요청 데이터를 캐시에서 조회합니다.",
            "contributions": [
                {
                    "title": "캐시 정책 적용",
                    "description": "기본 TTL을 30분으로 설정했습니다.",
                    "metrics": ["TTL 30분"],
                }
            ],
            "troubleshooting": [
                {
                    "title": "TTL 누락 해결",
                    "tags": ["Redis", "TTL"],
                    "situation": "TTL 설정이 누락되었습니다.",
                    "action": "기본 TTL을 적용했습니다.",
                    "result": "모든 캐시에 만료 정책을 적용했습니다.",
                }
            ],
            "retrospective": {
                "technicalGrowth": "캐시 만료 정책의 중요성을 학습했습니다.",
                "collaboration": "자료에서 확인되지 않음",
                "futureRoadmap": "제안: 캐시 적중률을 측정합니다.",
            },
            "missingEvidence": ["협업 과정은 자료에서 확인되지 않습니다."],
        },
    )

    response = TestClient(app).post(
        "/portfolio/report",
        json={
            "projectId": 7,
            "projectName": "AI 기반 프로젝트 검색 및 아카이빙 플랫폼",
            "period": "2025.09 ~ 2025.12",
            "teamSize": "5명",
            "role": "백엔드 리드 및 AI 파이프라인",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["projectName"] == "AI 기반 프로젝트 검색 및 아카이빙 플랫폼"
    assert body["executiveSummary"]["teamSize"] == "5명"
    assert body["techStack"][0]["category"] == "ARCHITECTURE"
    assert body["contributions"][0]["metrics"] == ["TTL 30분"]
    assert body["troubleshooting"][0]["action"] == "기본 TTL을 적용했습니다."
    assert body["retrospective"]["futureRoadmap"].startswith("제안:")
    assert body["citations"][0]["artifactId"] == 21
    assert body["generatedAt"].endswith("Z")


def test_portfolio_report_without_documents_exposes_missing_evidence() -> None:
    report = services.generate_portfolio_report(
        project_name="프로젝트",
        period=None,
        team_size=None,
        role=None,
        retrieved_documents=[],
    )

    assert report["techStack"] == []
    assert report["troubleshooting"] == []
    assert report["executiveSummary"]["period"] == "자료에서 확인되지 않음"
    assert report["missingEvidence"] == ["프로젝트 자료를 먼저 등록해 주세요."]
