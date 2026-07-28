from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

class DocumentRequest(BaseModel):
    """프로젝트 자료 등록 요청 형식"""
    user_id: int
    project_id: int
    project_name: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_type: str = Field(
        min_length=1,
        description="upload, github, google_drive 등",
    )
    source_url: Optional[str] = None
    text: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """프로젝트 Q&A 요청 형식"""
    user_id: int
    question: str = Field(min_length=1)
    project_id: Optional[int] = None
    top_k: int = Field(default=5, ge=1, le=10)
    answer_mode: str = Field(
        default="general",
        pattern="^(general|portfolio|interview)$",
    )


class DeleteSourceRequest(BaseModel):
    """기존 Chunk 삭제 요청 형식"""
    user_id: int
    project_id: int
    source_name: str
