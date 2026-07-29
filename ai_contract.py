from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from schemas import ChatRequest
from services import (
    SimpleVectorStore,
    create_embeddings,
    generate_answer,
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
    chunks: list[Chunk]


class QuestionRequest(BaseModel):
    projectId: int
    question: str = Field(min_length=1)


class SummaryRequest(BaseModel):
    projectId: int
    since: datetime


class DeleteArtifactsRequest(BaseModel):
    projectId: int
    artifactIds: list[int] = Field(min_length=1)


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
) -> dict[str, Any]:
    chat_request = ChatRequest(
        user_id=0,
        project_id=project_id,
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
        f"{request.since.isoformat()} 이후의 커밋과 회의록, 주요 진행 상황을 요약해줘.",
        occurred_since=request.since,
    )
    return {"summary": result["answer"]}


@router.post("/interview")
def interview(request: QuestionRequest) -> dict[str, Any]:
    return _answer(request.projectId, request.question, mode="interview")
