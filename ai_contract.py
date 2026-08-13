from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from schemas import ChatRequest
from services import (
    SimpleVectorStore,
    create_embeddings,
    generate_answer,
    generate_career_interview_questions,
    generate_career_star,
    generate_portfolio_report,
    retrieve_portfolio_context,
    retrieve_project_context,
)

router = APIRouter(tags=["AI server contract"])


class Chunk(BaseModel):
    artifactId: int
    type: Literal["COMMIT", "CODE", "DOC", "MEETING"]
    title: str
    path: Optional[str] = None
    url: Optional[str] = None
    author: Optional[str] = None
    occurredAt: Optional[datetime] = None
    seq: int = 0
    text: str = Field(min_length=1)


class IndexRequest(BaseModel):
    projectId: int
    projectName: Optional[str] = Field(default=None, min_length=1)
    chunks: list[Chunk]


class QuestionRequest(BaseModel):
    projectId: int
    question: str = Field(min_length=1)


class SummaryRequest(BaseModel):
    projectId: int
    projectName: Optional[str] = Field(default=None, min_length=1)
    since: datetime


class DeleteArtifactsRequest(BaseModel):
    projectId: int
    artifactIds: list[int] = Field(min_length=1)


class CareerStarRequest(BaseModel):
    projectId: int
    jobRole: str = Field(min_length=1)
    question: str = Field(min_length=1)


class CareerInterviewQuestionsRequest(BaseModel):
    projectId: int
    jobRole: str = Field(min_length=1)
    questionCount: int = Field(default=3, ge=1, le=5)


class PortfolioReportRequest(BaseModel):
    projectId: int
    projectName: Optional[str] = Field(default=None, min_length=1)
    period: Optional[str] = Field(default=None, min_length=1)
    teamSize: Optional[str] = Field(default=None, min_length=1)
    role: Optional[str] = Field(default=None, min_length=1)


class Citation(BaseModel):
    artifactId: int
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: str


class CareerStarSections(BaseModel):
    situation: str
    task: str
    action: str
    result: str


class CareerStarResponse(BaseModel):
    jobRole: str
    question: str
    star: CareerStarSections
    finalAnswer: str
    missingEvidence: list[str]
    citations: list[Citation]


class CareerFollowUp(BaseModel):
    question: str = Field(min_length=1)
    recommendedAnswer: str = Field(min_length=1)


class CareerInterviewQuestion(BaseModel):
    category: str = Field(min_length=1)
    likelihood: Literal["HIGH", "MEDIUM", "LOW"]
    question: str = Field(min_length=1)
    modelAnswer: str = Field(min_length=1)
    checkpoints: list[str] = Field(min_length=2, max_length=4)
    followUps: list[CareerFollowUp] = Field(min_length=2, max_length=2)
    citations: list[Citation]


class CareerInterviewQuestionsResponse(BaseModel):
    questions: list[CareerInterviewQuestion]


class PortfolioExecutiveSummary(BaseModel):
    servicePurpose: str
    targetUsers: str
    period: str
    teamSize: str
    role: str


class PortfolioTechStack(BaseModel):
    name: str
    category: Literal[
        "FRONTEND",
        "BACKEND",
        "DATABASE",
        "AI_ML",
        "ARCHITECTURE",
        "DEVOPS",
        "OTHER",
    ]
    reason: str


class PortfolioContribution(BaseModel):
    title: str
    description: str
    metrics: list[str] = Field(min_length=1, max_length=3)


class PortfolioTroubleshooting(BaseModel):
    title: str
    tags: list[str]
    situation: str
    action: str
    result: str


class PortfolioRetrospective(BaseModel):
    technicalGrowth: str
    collaboration: str
    futureRoadmap: str


class PortfolioReportResponse(BaseModel):
    projectId: int
    projectName: str
    generatedAt: datetime
    oneLineSummary: str
    executiveSummary: PortfolioExecutiveSummary
    techStack: list[PortfolioTechStack]
    systemArchitecture: str
    dataPipeline: str
    contributions: list[PortfolioContribution] = Field(max_length=5)
    troubleshooting: list[PortfolioTroubleshooting]
    retrospective: PortfolioRetrospective
    missingEvidence: list[str]
    citations: list[Citation]


def _citations(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[int] = set()
    for document in documents:
        metadata = document.get("metadata", {})
        artifact_id = metadata.get("artifact_id")
        if artifact_id is None or artifact_id in seen:
            continue
        seen.add(artifact_id)
        citations.append(
            {
                "artifactId": artifact_id,
                "title": metadata.get("source_name"),
                "url": metadata.get("source_url"),
                "snippet": document.get("text", "")[:240],
            }
        )
    return citations


@router.post("/index")
def index_documents(request: IndexRequest) -> dict[str, Any]:
    """Spring 백엔드가 파싱·청킹한 자료를 임베딩하고 프로젝트별로 저장한다."""
    if not request.chunks:
        return {"indexed": 0, "techStack": []}

    embeddings = create_embeddings([chunk.text for chunk in request.chunks])
    if len(embeddings) != len(request.chunks):
        raise HTTPException(500, "임베딩 결과 수가 청크 수와 일치하지 않습니다.")

    documents: list[dict[str, Any]] = []
    detected: set[str] = set()
    tech_keywords = {
        "Python", "Java", "Spring", "FastAPI", "React", "Vue", "TypeScript",
        "JavaScript", "PostgreSQL", "MySQL", "Redis", "Docker", "AWS",
        "Kubernetes", "GitHub", "Notion",
    }
    for chunk, embedding in zip(request.chunks, embeddings):
        detected.update(word for word in tech_keywords if word.lower() in chunk.text.lower())
        documents.append(
            {
                "id": str(uuid4()),
                "text": chunk.text,
                "embedding": embedding,
                "metadata": {
                    "user_id": 0,
                    "project_id": request.projectId,
                    "project_name": request.projectName or f"Project {request.projectId}",
                    "artifact_id": chunk.artifactId,
                    "source_name": chunk.title,
                    "source_type": chunk.type,
                    "source_url": chunk.url,
                    "path": chunk.path,
                    "author": chunk.author,
                    "occurred_at": chunk.occurredAt.isoformat() if chunk.occurredAt else None,
                    "chunk_index": chunk.seq,
                },
            }
        )

    store = SimpleVectorStore()
    artifact_ids = {chunk.artifactId for chunk in request.chunks}
    store.replace_artifacts(request.projectId, artifact_ids, documents)
    return {"indexed": len(documents), "techStack": sorted(detected)}


@router.post("/index/delete")
def delete_artifacts(request: DeleteArtifactsRequest) -> dict[str, int]:
    deleted = SimpleVectorStore().delete_artifacts(
        request.projectId,
        set(request.artifactIds),
    )
    return {"deleted": deleted}


@router.delete("/index/projects/{project_id}")
def delete_project_index(project_id: int) -> dict[str, int]:
    return {"deleted": SimpleVectorStore().delete_project(project_id)}


def _answer(
    project_id: int,
    question: str,
    mode: str = "general",
    occurred_since: Optional[datetime] = None,
    project_name: Optional[str] = None,
) -> dict[str, Any]:
    chat_request = ChatRequest(
        user_id=0,
        project_id=project_id,
        project_name=project_name,
        question=question,
        answer_mode=mode,
        top_k=8,
    )
    docs = retrieve_project_context(chat_request, occurred_since=occurred_since)
    answer = generate_answer(chat_request, docs)
    return {"answer": answer, "citations": _citations(docs)}


@router.post("/chat")
def contract_chat(request: QuestionRequest) -> dict[str, Any]:
    return _answer(request.projectId, request.question)


@router.post("/summary")
def summary(request: SummaryRequest) -> dict[str, str]:
    result = _answer(
        request.projectId,
        f"{request.since.date().isoformat()} 이후의 커밋과 회의록, 주요 진행 상황을 요약해줘.",
        mode="summary",
        occurred_since=request.since,
        project_name=request.projectName,
    )
    return {"summary": result["answer"]}


@router.post("/interview")
def interview(request: QuestionRequest) -> dict[str, Any]:
    return _answer(request.projectId, request.question, mode="interview")


@router.post("/career/star", response_model=CareerStarResponse)
def career_star(request: CareerStarRequest) -> dict[str, Any]:
    chat_request = ChatRequest(
        user_id=0,
        project_id=request.projectId,
        question=f"{request.jobRole} 지원 자기소개서 문항: {request.question}",
        answer_mode="portfolio",
        top_k=8,
    )
    documents = retrieve_project_context(chat_request)
    generated = generate_career_star(request.jobRole, request.question, documents)
    return {
        "jobRole": request.jobRole,
        "question": request.question,
        **generated,
        "citations": _citations(documents),
    }


@router.post(
    "/career/interview-questions",
    response_model=CareerInterviewQuestionsResponse,
)
def career_interview_questions(
    request: CareerInterviewQuestionsRequest,
) -> dict[str, Any]:
    chat_request = ChatRequest(
        user_id=0,
        project_id=request.projectId,
        question=(
            f"{request.jobRole} 직무 기술 면접: 프로젝트 아키텍처, "
            "기술 선택, 성능 개선, 장애 대응"
        ),
        answer_mode="interview",
        top_k=8,
    )
    documents = retrieve_project_context(chat_request)
    citations = _citations(documents)
    questions = generate_career_interview_questions(
        request.jobRole,
        request.questionCount,
        documents,
    )
    return {
        "questions": [
            {**question, "citations": citations}
            for question in questions
        ]
    }


@router.post("/portfolio/report", response_model=PortfolioReportResponse)
def portfolio_report(request: PortfolioReportRequest) -> dict[str, Any]:
    documents = retrieve_portfolio_context(request.projectId)
    indexed_project_name = next(
        (
            document.get("metadata", {}).get("project_name")
            for document in documents
            if document.get("metadata", {}).get("project_name")
        ),
        None,
    )
    project_name = (
        request.projectName
        or indexed_project_name
        or f"Project {request.projectId}"
    )
    report = generate_portfolio_report(
        project_name=project_name,
        period=request.period,
        team_size=request.teamSize,
        role=request.role,
        retrieved_documents=documents,
    )
    return {
        "projectId": request.projectId,
        "projectName": project_name,
        "generatedAt": datetime.now(timezone.utc),
        **report,
        "citations": _citations(documents),
    }
