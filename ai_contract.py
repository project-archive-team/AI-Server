from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from schemas import ChatRequest
from services import (
    SimpleVectorStore,
    build_sources,
    create_embeddings,
    generate_answer,
    retrieve_project_context,
)

router = APIRouter(tags=["AI server contract"])


class Chunk(BaseModel):
    artifactId: int
    type: Literal["COMMIT", "CODE", "DOC", "MEETING"]
    title: str
    path: str | None = None
    url: str | None = None
    author: str | None = None
    occurredAt: datetime | None = None
    seq: int = 0
    text: str = Field(min_length=1)


class IndexRequest(BaseModel):
    projectId: int
    chunks: list[Chunk]


class QuestionRequest(BaseModel):
    projectId: int
    question: str = Field(min_length=1)


class SummaryRequest(BaseModel):
    projectId: int
    since: datetime


def _citations(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "artifactId": source.get("artifact_id"),
            "title": source.get("source_name"),
            "url": source.get("source_url"),
            "snippet": source.get("snippet", ""),
        }
        for source in sources
    ]


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
                    "project_name": f"Project {request.projectId}",
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
    store.documents = [
        doc for doc in store.documents
        if not (
            doc.get("metadata", {}).get("project_id") == request.projectId
            and doc.get("metadata", {}).get("artifact_id") in artifact_ids
        )
    ]
    store.add_documents(documents)
    return {"indexed": len(documents), "techStack": sorted(detected)}


def _answer(project_id: int, question: str, mode: str = "general") -> dict[str, Any]:
    chat_request = ChatRequest(
        user_id=0,
        project_id=project_id,
        question=question,
        answer_mode=mode,
        top_k=8,
    )
    docs = retrieve_project_context(chat_request)
    answer = generate_answer(chat_request, docs)
    sources = build_sources(docs)
    for source, doc in zip(sources, docs):
        source["artifact_id"] = doc.get("metadata", {}).get("artifact_id")
        source["snippet"] = doc.get("text", "")[:240]
    return {"answer": answer, "citations": _citations(sources)}


@router.post("/chat")
def contract_chat(request: QuestionRequest) -> dict[str, Any]:
    return _answer(request.projectId, request.question)


@router.post("/summary")
def summary(request: SummaryRequest) -> dict[str, str]:
    result = _answer(
        request.projectId,
        f"{request.since.isoformat()} 이후의 커밋과 회의록, 주요 진행 상황을 요약해줘.",
    )
    return {"summary": result["answer"]}


@router.post("/interview")
def interview(request: QuestionRequest) -> dict[str, Any]:
    return _answer(request.projectId, request.question, mode="interview")
