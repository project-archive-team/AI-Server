from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import numpy as np
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from google.genai import types

# 프로젝트 내의 다른 모듈 불러오기
from config import client, GEMINI_EMBEDDING_MODEL, GEMINI_CHAT_MODEL, STORE_PATH
from schemas import DocumentRequest, ChatRequest


# 구조화된 출력을 위한 스키마 정의
class InitialQuestionsSchema(BaseModel):
    recommended_questions: list[str]


class CareerStarSectionsSchema(BaseModel):
    situation: str = Field(min_length=1)
    task: str = Field(min_length=1)
    action: str = Field(min_length=1)
    result: str = Field(min_length=1)


class CareerStarGenerationSchema(BaseModel):
    star: CareerStarSectionsSchema
    finalAnswer: str = Field(min_length=1)
    missingEvidence: list[str]


class CareerFollowUpSchema(BaseModel):
    question: str = Field(min_length=1)
    recommendedAnswer: str = Field(min_length=1)


class CareerInterviewQuestionSchema(BaseModel):
    category: str = Field(min_length=1)
    likelihood: Literal["HIGH", "MEDIUM", "LOW"]
    question: str = Field(min_length=1)
    modelAnswer: str = Field(min_length=1)
    checkpoints: list[str] = Field(min_length=2, max_length=4)
    followUps: list[CareerFollowUpSchema] = Field(min_length=2, max_length=2)


class CareerInterviewQuestionsGenerationSchema(BaseModel):
    questions: list[CareerInterviewQuestionSchema] = Field(min_length=1, max_length=5)


class PortfolioExecutiveSummarySchema(BaseModel):
    servicePurpose: str = Field(min_length=1)
    targetUsers: str = Field(min_length=1)
    period: str = Field(min_length=1)
    teamSize: str = Field(min_length=1)
    role: str = Field(min_length=1)


class PortfolioTechStackSchema(BaseModel):
    name: str = Field(min_length=1)
    category: Literal[
        "FRONTEND",
        "BACKEND",
        "DATABASE",
        "AI_ML",
        "ARCHITECTURE",
        "DEVOPS",
        "OTHER",
    ]
    reason: str = Field(min_length=1)


class PortfolioContributionSchema(BaseModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    metrics: list[str] = Field(max_length=3)


class PortfolioTroubleshootingSchema(BaseModel):
    title: str = Field(min_length=1)
    tags: list[str] = Field(max_length=6)
    situation: str = Field(min_length=1)
    action: str = Field(min_length=1)
    result: str = Field(min_length=1)


class PortfolioRetrospectiveSchema(BaseModel):
    technicalGrowth: str = Field(min_length=1)
    collaboration: str = Field(min_length=1)
    futureRoadmap: str = Field(min_length=1)


class PortfolioReportGenerationSchema(BaseModel):
    oneLineSummary: str = Field(min_length=1)
    executiveSummary: PortfolioExecutiveSummarySchema
    techStack: list[PortfolioTechStackSchema] = Field(max_length=12)
    systemArchitecture: str = Field(min_length=1)
    dataPipeline: str = Field(min_length=1)
    contributions: list[PortfolioContributionSchema] = Field(max_length=6)
    troubleshooting: list[PortfolioTroubleshootingSchema] = Field(max_length=5)
    retrospective: PortfolioRetrospectiveSchema
    missingEvidence: list[str]


# ------------------------------------------------------------
# 3-1. 문서 Chunking & 임베딩 & 유사도 계산
# ------------------------------------------------------------

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    if not text or not text.strip():
        return []
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("부적절한 chunk_size 또는 chunk_overlap 값입니다.")

    cleaned_text = " ".join(text.split())
    chunks: list[str] = []
    start = 0

    while start < len(cleaned_text):
        end = start + chunk_size
        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def create_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Embedding할 텍스트가 비어 있습니다.")
    
    normalized_text = text.replace("\n", " ").strip()
    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=normalized_text,
    )
    return response.embeddings[0].values


def create_embeddings(texts: list[str]) -> list[list[float]]:
    normalized_texts = [
        text.replace("\n", " ").strip()
        for text in texts
        if text and text.strip()
    ]
    if not normalized_texts:
        return []

    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=normalized_texts,
    )
    return [emb.values for emb in response.embeddings]


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    a = np.asarray(vector_a, dtype=np.float32)
    b = np.asarray(vector_b, dtype=np.float32)

    if a.shape != b.shape:
        raise ValueError("두 Embedding 벡터의 차원이 다릅니다.")

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# ------------------------------------------------------------
# 3-2. 로컬 JSON 기반 Vector Store
# ------------------------------------------------------------

_STORE_LOCK = threading.RLock()


class SimpleVectorStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or STORE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _STORE_LOCK:
            self.documents = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Vector Store JSON 파일 손상: {self.path}") from error
        
        if not isinstance(data, list):
            raise RuntimeError("Vector Store 데이터는 리스트 형식이어야 합니다.")
        return data

    def _save(self) -> None:
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self.documents, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, self.path)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return
        with _STORE_LOCK:
            self.documents = self._load()
            self.documents.extend(documents)
            self._save()

    def replace_artifacts(
        self,
        project_id: int,
        artifact_ids: set[int],
        documents: list[dict[str, Any]],
    ) -> None:
        """Replace every indexed chunk for the supplied artifacts atomically."""
        with _STORE_LOCK:
            self.documents = [
                document
                for document in self._load()
                if not (
                    document.get("metadata", {}).get("project_id") == project_id
                    and document.get("metadata", {}).get("artifact_id") in artifact_ids
                )
            ]
            self.documents.extend(documents)
            self._save()

    def delete_source(self, user_id: int, project_id: int, source_name: str) -> int:
        with _STORE_LOCK:
            current = self._load()
            self.documents = [
                doc for doc in current
                if not (
                    doc.get("metadata", {}).get("user_id") == user_id
                    and doc.get("metadata", {}).get("project_id") == project_id
                    and doc.get("metadata", {}).get("source_name") == source_name
                )
            ]
            deleted_count = len(current) - len(self.documents)
            if deleted_count:
                self._save()
            return deleted_count

    def delete_artifacts(self, project_id: int, artifact_ids: set[int]) -> int:
        with _STORE_LOCK:
            current = self._load()
            self.documents = [
                document
                for document in current
                if not (
                    document.get("metadata", {}).get("project_id") == project_id
                    and document.get("metadata", {}).get("artifact_id") in artifact_ids
                )
            ]
            deleted_count = len(current) - len(self.documents)
            if deleted_count:
                self._save()
            return deleted_count

    def delete_project(self, project_id: int) -> int:
        with _STORE_LOCK:
            current = self._load()
            self.documents = [
                document
                for document in current
                if document.get("metadata", {}).get("project_id") != project_id
            ]
            deleted_count = len(current) - len(self.documents)
            if deleted_count:
                self._save()
            return deleted_count

    def search(
        self,
        query_embedding: list[float],
        user_id: int,
        project_id: Optional[int] = None,
        top_k: int = 5,
        occurred_since: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        with _STORE_LOCK:
            documents = self._load()
        scored_documents: list[dict[str, Any]] = []
        for document in documents:
            metadata = document.get("metadata", {})
            if metadata.get("user_id") != user_id:
                continue
            if project_id is not None and metadata.get("project_id") != project_id:
                continue
            if occurred_since is not None:
                occurred_at = metadata.get("occurred_at")
                if not occurred_at:
                    continue
                try:
                    if datetime.fromisoformat(occurred_at) < occurred_since:
                        continue
                except (TypeError, ValueError):
                    continue

            embedding = document.get("embedding")
            if not embedding:
                continue

            score = cosine_similarity(query_embedding, embedding)
            scored_documents.append({**document, "score": score})

        scored_documents.sort(key=lambda item: item["score"], reverse=True)
        return scored_documents[:top_k]


# ------------------------------------------------------------
# 3-3. 비즈니스 로직 조율 및 질문 생성 함수들
# ------------------------------------------------------------

def generate_initial_questions(user_id: int, project_id: int, answer_mode: str = "general") -> list[str]:
    """선택한 프로젝트와 '선택된 답변 모드'에 맞추어 성격이 다른 연관 추천 질문 3개를 생성합니다."""
    vector_store = SimpleVectorStore()
    
    # 1. 프로젝트 문서 수집
    project_docs = [
        doc for doc in vector_store.documents
        if doc.get("metadata", {}).get("user_id") == user_id
        and doc.get("metadata", {}).get("project_id") == project_id
    ]
    
    project_name = project_docs[0].get("metadata", {}).get("project_name", "선택된 프로젝트") if project_docs else "선택된 프로젝트"

    # 모드별 기본 방어용 질문 세트 정의
    if answer_mode == "portfolio":
        default_questions = [
            "프로젝트에서 본인의 역할과 기여를 설명해주세요",
            "협업 중 갈등을 어떻게 해결했나요?",
            "해당 기술스택을 선택한 이유는 무엇인가요?"
        ]
        mode_hint = "자기소개서나 포트폴리오 서술용 질문 (역할, 기여도, 협업, 기술 스택 도입 이유 등)"
    elif answer_mode == "interview":
        default_questions = [
            "DB 스키마 설계 경험을 설명해주세요",
            "이 프로젝트의 기술적 난관은 무엇이었나요?",
            "성능 최적화를 위해 무엇을 했나요?"
        ]
        mode_hint = "실전 기술 면접용 질문 (기술적 난관, 성능 최적화, 아키텍처 한계, 압박 질문 등)"
    else:
        default_questions = [
            "이 프로젝트의 핵심 기능은 무엇인가요?",
            "프로젝트의 주요 타겟 사용자는 누구인가요?",
            "개발 프로세스와 전반적인 흐름을 요약해 주세요."
        ]
        mode_hint = "일반적인 프로젝트 내용 파악용 질문"

    if not project_docs:
        return default_questions
        
    # 분석용 컨텍스트 조립
    sample_texts = [doc.get("text", "") for doc in project_docs[:5]]
    project_context = "\n\n".join(sample_texts)

    # 2. Gemini에게 모드 맞춤형 질문 도출 지시
    prompt = f"""
당신은 사용자가 선택한 프로젝트 정보를 분석하여, 현재 지정된 목적({answer_mode})에 어울리는 추천 질문 리스트를 제안하는 전문가입니다.

[분석 대상 프로젝트]
- 프로젝트명: {project_name}

[프로젝트 문서 내용 일부]
{project_context}

[추천 질문 생성 지침]
현재 사용자는 '{answer_mode}' 목적으로 질문 대기 중입니다.
이 목적에 맞게 아래 가이드라인을 준수하여 이 프로젝트에 특화된 구체적인 '추천 질문' 딱 3개를 생성해 주세요.

- 목적별 성격 규칙:
  * portfolio 일 때: {mode_hint}
  * interview 일 때: {mode_hint}
  * general 일 때: {mode_hint}

주의사항:
1. 단순히 범용적인 질문이 아니라, 제공된 프로젝트의 구체적인 기술이나 내용이 녹아 있는 날카로운 질문이어야 합니다.
2. 리스트 형태의 JSON으로 반환해야 합니다.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InitialQuestionsSchema,
                temperature=0.4,
            )
        )
        result = json.loads(response.text)
        return result.get("recommended_questions", [])[:3]
    except Exception:
        return default_questions
    
def get_unique_projects(user_id: int, answer_mode: str = "general") -> list[dict[str, Any]]:
    """사용자가 등록한 프로젝트 목록과, 선택한 모드에 따른 맞춤 추천 질문 3개를 함께 반환합니다."""
    vector_store = SimpleVectorStore()
    
    seen_project_ids = set()
    projects = []
    
    for doc in vector_store.documents:
        metadata = doc.get("metadata", {})
        u_id = metadata.get("user_id")
        p_id = metadata.get("project_id")
        p_name = metadata.get("project_name", "이름 없음")
        
        if u_id == user_id and p_id is not None:
            if p_id not in seen_project_ids:
                seen_project_ids.add(p_id)
                projects.append({
                    "project_id": p_id,
                    "project_name": p_name,
                    "recommended_questions": []
                })
                
    # 질문 생성 시 answer_mode를 함께 넘겨줍니다.
    for proj in projects:
        p_id = proj["project_id"]
        proj["recommended_questions"] = generate_initial_questions(
            user_id=user_id, 
            project_id=p_id,
            answer_mode=answer_mode
        )
                
    return projects


def ingest_project_document(request: DocumentRequest) -> int:
    chunks = split_text(text=request.text)
    if not chunks:
        raise ValueError("등록할 수 있는 텍스트가 없습니다.")

    embeddings = create_embeddings(chunks)
    if len(chunks) != len(embeddings):
        raise RuntimeError("Chunk 수와 Embedding 수가 일치하지 않습니다.")

    documents: list[dict[str, Any]] = []
    for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        documents.append({
            "id": str(uuid4()),
            "text": chunk,
            "embedding": embedding,
            "metadata": {
                "user_id": request.user_id,
                "project_id": request.project_id,
                "project_name": request.project_name,
                "source_name": request.source_name,
                "source_type": request.source_type,
                "source_url": request.source_url,
                "chunk_index": chunk_index,
            },
        })

    vector_store = SimpleVectorStore()
    vector_store.add_documents(documents)
    return len(documents)


def retrieve_project_context(
    request: ChatRequest,
    occurred_since: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    query_embedding = create_embedding(request.question)
    vector_store = SimpleVectorStore()
    return vector_store.search(
        query_embedding=query_embedding,
        user_id=request.user_id,
        project_id=request.project_id,
        top_k=request.top_k,
        occurred_since=occurred_since,
    )


def retrieve_portfolio_context(
    project_id: int,
    user_id: int = 0,
    max_documents: int = 20,
) -> list[dict[str, Any]]:
    """포트폴리오 각 섹션에 필요한 근거를 다중 관점으로 검색하고 중복을 제거한다."""
    queries = [
        "프로젝트 목적 대상 사용자 기간 팀 구성 담당 역할 핵심 기능",
        "기술 스택 선택 이유 시스템 아키텍처 데이터 처리 흐름",
        "구현 기능 담당 업무 핵심 기여 성능 개선 정량 성과",
        "장애 문제 원인 해결 과정 트러블슈팅 결과 회고 협업 개선 계획",
    ]
    merged: list[dict[str, Any]] = []
    seen: set[Any] = set()

    for query in queries:
        request = ChatRequest(
            user_id=user_id,
            project_id=project_id,
            question=query,
            answer_mode="portfolio",
            top_k=8,
        )
        for document in retrieve_project_context(request):
            metadata = document.get("metadata", {})
            key = document.get("id") or (
                metadata.get("artifact_id"),
                metadata.get("chunk_index"),
                document.get("text", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(document)
            if len(merged) >= max_documents:
                return merged

    return merged


def build_context(retrieved_documents: list[dict[str, Any]]) -> str:
    context_parts: list[str] = []
    for index, document in enumerate(retrieved_documents, start=1):
        metadata = document.get("metadata", {})
        context_parts.append(
            f"[근거 자료 {index}]\n프로젝트명: {metadata.get('project_name', '알 수 없음')}\n파일명: {metadata.get('source_name', '알 수 없음')}\n자료 유형: {metadata.get('source_type', '알 수 없음')}\n원본 링크: {metadata.get('source_url') or '없음'}\n검색 유사도: {document.get('score', 0):.4f}\n\n내용:\n{document.get('text', '')}".strip()
        )
    return "\n\n".join(context_parts)


def get_mode_instruction(answer_mode: str) -> str:
    if answer_mode == "portfolio":
        return "## 프로젝트 분석\n질문에 대한 핵심 내용을 설명합니다.\n\n## 포트폴리오용 문장\n전문적인 문장 2~4개를 작성합니다.\n\n## 강조하면 좋은 역량\n기술, 문제 해결 역량을 정리합니다.\n\n## 보완하면 좋은 정보\n부족한 수치나 역할을 알려줍니다."
    
    if answer_mode == "interview":
        return """
## 핵심 답변
면접에서 말할 수 있는 형태로 질문에 대해 명확히 답변합니다.

## 30초 답변 예시
실제 면접장에서 대화하듯 자연스럽게 말할 수 있는 구어체 문장을 작성합니다.

## 예상 꼬리 질문과 추천 답변
이 답변을 들은 면접관이 추가로 물어볼 만한 날카로운 질문을 정확히 3개 작성합니다.
각 질문 바로 아래에 지원자가 말할 수 있는 추천 답변을 함께 작성합니다.

다음 형식을 반드시 지킵니다.

### 1. 예상 질문
질문 내용을 작성합니다.

**추천 답변**
프로젝트 자료에 근거한 2~4문장의 구어체 답변을 작성합니다.

추천 답변에도 없는 사실이나 수치를 만들지 않습니다. 답변에 필요한 정보가 자료에
부족하면 어떤 경험이나 수치를 추가로 준비해야 하는지 솔직하게 안내합니다.
""".strip()

    return "## 핵심 답변\n직접 답변을 작성합니다.\n\n## 프로젝트에서 확인된 내용\n역할, 기술, 해결 과정을 요약합니다.\n\n## 강점\n드러나는 강점을 요약합니다.\n\n## 더 보완하면 좋은 점\n추가 보완점을 알려줍니다."


def generate_answer(request: ChatRequest, retrieved_documents: list[dict[str, Any]]) -> str:
    if not retrieved_documents:
        return "질문과 관련된 프로젝트 자료를 찾지 못했습니다.\n\n자료를 먼저 등록해 주세요."

    context = build_context(retrieved_documents)
    mode_instruction = get_mode_instruction(request.answer_mode)

    instructions = f"""당신은 개발자의 프로젝트 기록을 분석하는 프로젝트 아카이빙 AI 어시스턴트입니다.
반드시 제공된 프로젝트 자료만 근거로 답변하세요.

중요 규칙:
1. 답변을 시작할 때 "제공된 자료에 따르면", "등록된 자료는 ~를 보여줍니다"와 같이 자료를 언급하는 서론을 절대 쓰지 마세요.
2. 질문에 대해 즉시 "저는 ~ 프로젝트에서 ~를 담당했습니다"와 같이 사용자의 관점에서 직접적이고 두괄식으로 답변을 시작하세요.
3. 없는 사실을 지어내지 마세요.
4. 불확실한 내용은 "등록된 자료에서 확인하기 어렵습니다"라고 답하세요.
5. 한국어 Markdown으로 작성하세요.
6. 같은 내용을 반복하지 마세요.   

{mode_instruction}""".strip()

    prompt = f"아래 자료를 근거로 답하세요.\n\n{context}\n\n질문: {request.question}\n모드: {request.answer_mode}\n\n마지막에는 아래 형식으로 참고자료 출처를 표시하세요.\n## 참고 자료\n- 프로젝트명 / 파일명"

    response = client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            temperature=0.3,
        )
    )
    return response.text


def generate_career_star(
    job_role: str,
    question: str,
    retrieved_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """프로젝트 근거만 사용해 프론트에서 바로 그릴 수 있는 STAR 답변을 생성한다."""
    if not retrieved_documents:
        unavailable = "등록된 프로젝트 자료가 없어 답변을 생성할 수 없습니다."
        return {
            "star": {
                "situation": unavailable,
                "task": unavailable,
                "action": unavailable,
                "result": unavailable,
            },
            "finalAnswer": unavailable,
            "missingEvidence": ["프로젝트 자료를 먼저 등록해 주세요."],
        }

    context = build_context(retrieved_documents)
    system_instruction = """
당신은 개발자의 프로젝트 기록을 취업용 자기소개서 답변으로 정리하는 전문가입니다.
반드시 제공된 프로젝트 자료에 명시된 사실만 사용합니다.

규칙:
1. 자료에 없는 기술, 역할, 장애 원인, 성과, 수치, 전후 비교를 절대 만들지 않습니다.
2. "크게 개선", "대폭 감소"처럼 측정 근거가 필요한 표현은 자료에 수치나 명시적 평가가 있을 때만 사용합니다.
3. 지원 직무에 맞게 표현을 다듬되 프로젝트 사실 자체를 바꾸지 않습니다.
4. STAR 각 항목은 서로 중복하지 않고 지원자 관점의 한국어 문장으로 작성합니다.
5. 질문에 답하는 데 필요한 역할이나 성과 수치가 자료에 없으면 추측하지 말고 missingEvidence에 구체적으로 적습니다.
6. finalAnswer는 STAR 내용을 자연스럽게 연결한 완성형 답변이며 새로운 사실을 추가하지 않습니다.
7. 출처나 인용 정보는 출력하지 않습니다. 인용은 서버가 별도로 결합합니다.
""".strip()
    prompt = f"""
[지원 직무]
{job_role}

[자기소개서 문항]
{question}

[프로젝트 근거 자료]
{context}

위 근거만 사용해 Situation, Task, Action, Result와 완성형 답변을 작성하세요.
""".strip()

    response = client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=CareerStarGenerationSchema,
            temperature=0.2,
        ),
    )
    return CareerStarGenerationSchema.model_validate_json(response.text).model_dump()


def generate_career_interview_questions(
    job_role: str,
    question_count: int,
    retrieved_documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """프로젝트 아키텍처에 근거한 면접 질문 카드 목록을 생성한다."""
    if not retrieved_documents:
        return []

    context = build_context(retrieved_documents)
    system_instruction = f"""
당신은 개발 프로젝트 기반 실전 기술 면접 질문을 설계하는 면접관입니다.
반드시 제공된 프로젝트 자료에 명시된 사실만 사용해 정확히 {question_count}개의 질문 카드를 만듭니다.

규칙:
1. 자료에 없는 기술, 역할, 장애, 성과 수치, 응답 시간, 데이터 규모를 절대 만들지 않습니다.
2. 질문은 지원 직무와 프로젝트의 실제 아키텍처, 기술 선택, 트러블슈팅에 연결합니다.
3. likelihood는 HIGH, MEDIUM, LOW 중 하나만 사용합니다.
4. modelAnswer는 지원자가 실제 면접에서 말할 수 있는 STAR 흐름의 한국어 답변으로 작성합니다.
5. 근거가 부족한 내용은 아는 척하지 말고 확인할 수 없다고 명시합니다.
6. checkpoints는 면접관이 평가할 핵심 요소를 2~4개 작성합니다.
7. followUps는 질문마다 정확히 2개 만들고, 각 질문에 근거 기반 recommendedAnswer를 제공합니다.
8. 추천 답변에도 원문에 없는 사실이나 수치를 추가하지 않습니다.
9. 출처나 인용 정보는 출력하지 않습니다. 인용은 서버가 별도로 결합합니다.
""".strip()
    prompt = f"""
[지원 직무]
{job_role}

[프로젝트 근거 자료]
{context}

지원 직무와 프로젝트에 특화된 실전 기술 면접 질문 카드를 생성하세요.
질문 카테고리는 아키텍처, 성능, 장애 대응, 데이터, 협업 중 실제 근거가 있는 영역에서 선택하세요.
""".strip()

    response = client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=CareerInterviewQuestionsGenerationSchema,
            temperature=0.2,
        ),
    )
    result = CareerInterviewQuestionsGenerationSchema.model_validate_json(response.text)
    return [question.model_dump() for question in result.questions[:question_count]]


def generate_portfolio_report(
    project_name: str,
    period: Optional[str],
    team_size: Optional[str],
    role: Optional[str],
    retrieved_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """아카이브 자료를 화면용 개발자 포트폴리오 리포트로 구조화한다."""
    unavailable = "자료에서 확인되지 않음"
    if not retrieved_documents:
        return {
            "oneLineSummary": unavailable,
            "executiveSummary": {
                "servicePurpose": unavailable,
                "targetUsers": unavailable,
                "period": period or unavailable,
                "teamSize": team_size or unavailable,
                "role": role or unavailable,
            },
            "techStack": [],
            "systemArchitecture": unavailable,
            "dataPipeline": unavailable,
            "contributions": [],
            "troubleshooting": [],
            "retrospective": {
                "technicalGrowth": unavailable,
                "collaboration": unavailable,
                "futureRoadmap": "근거 자료를 보완한 뒤 개선 계획을 작성해 주세요.",
            },
            "missingEvidence": ["프로젝트 자료를 먼저 등록해 주세요."],
        }

    context = build_context(retrieved_documents)
    supplied_metadata = (
        f"- 프로젝트명: {project_name}\n"
        f"- 진행 기간: {period or '미입력'}\n"
        f"- 팀 규모: {team_size or '미입력'}\n"
        f"- 담당 역할: {role or '미입력'}"
    )
    system_instruction = """
당신은 수집된 개발 산출물을 근거로 개발자 포트폴리오 리포트를 작성하는 전문가입니다.
결과는 채용 담당자와 기술 면접관이 빠르게 프로젝트의 가치, 설계 판단, 지원자의 기여와 문제 해결 능력을 파악할 수 있어야 합니다.

절대 규칙:
1. 제공된 프로젝트 메타데이터와 근거 자료에 명시된 사실만 사용합니다.
2. 자료에 없는 기술, 역할, 팀 규모, 기간, 장애 원인, 성능 수치, 전후 비교를 만들지 않습니다.
3. 수치가 없는 성과는 정성적으로만 쓰며 "대폭", "크게", "향상률" 같은 측정 표현을 사용하지 않습니다.
4. 서로 다른 문서의 사실을 하나의 사건처럼 임의로 합치지 않습니다.
5. 확인할 수 없는 필드는 "자료에서 확인되지 않음"이라고 쓰고 missingEvidence에도 보완 항목을 기록합니다.
6. techStack의 reason은 실제 도입 이유가 자료에 있을 때만 적고, 단순 사용 사실만 있으면 "사용 사실만 확인되며 선택 이유는 자료에서 확인되지 않음"이라고 씁니다.
7. metrics에는 원문에 명시된 정량 수치만 넣습니다. 근거 수치가 없으면 빈 배열을 반환합니다.
8. troubleshooting은 실제 문제 상황과 해결 행동이 함께 확인되는 사례만 작성합니다. 결과가 불명확하면 그대로 밝힙니다.
9. retrospective의 technicalGrowth와 collaboration은 기록에서 드러난 학습만 요약합니다.
10. futureRoadmap은 기존에 결정된 계획과 AI의 제안을 구분하여, 제안이라면 "제안:"으로 시작합니다.
11. 출처나 인용 정보는 출력하지 않습니다. 인용은 서버가 검색 문서에서 결합합니다.
""".strip()
    prompt = f"""
[백엔드가 제공한 프로젝트 메타데이터]
{supplied_metadata}

[프로젝트 근거 자료]
{context}

다음 화면 구성을 위한 한국어 포트폴리오 리포트를 작성하세요.
- 한 줄 요약
- 프로젝트 한 줄 요약(서비스 목적, 대상 사용자, 기간, 팀 규모, 담당 역할)
- 기술 스택 및 아키텍처(기술별 구분과 도입 이유, 시스템 아키텍처, 데이터 처리 흐름)
- 핵심 역할 및 기여
- 트러블슈팅 및 문제 해결 과정(Situation, Action, Result)
- 회고 및 배운 점(기술 성장, 협업 인사이트, 향후 개선점)
""".strip()

    response = client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=PortfolioReportGenerationSchema,
            temperature=0.2,
        ),
    )
    return PortfolioReportGenerationSchema.model_validate_json(response.text).model_dump()


def build_sources(retrieved_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()

    for document in retrieved_documents:
        metadata = document.get("metadata", {})
        key = (metadata.get("project_id"), metadata.get("source_name"), metadata.get("source_url"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "project_id": metadata.get("project_id"),
            "project_name": metadata.get("project_name"),
            "source_name": metadata.get("source_name"),
            "source_type": metadata.get("source_type"),
            "source_url": metadata.get("source_url"),
            "score": round(float(document.get("score", 0)), 4),
        })
    return sources

LOG_PATH = Path("data/rag_logs.json")

def log_rag_query(request: ChatRequest, retrieved_documents: list[dict[str, Any]]) -> None:
    """사용자의 질문과 그때 매칭된 최고/평균 스코어 등의 디버깅 로그를 로컬 JSON에 기록합니다."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 기존 로그 로드
    logs = []
    if LOG_PATH.exists():
        try:
            with LOG_PATH.open("r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
            
    # 스코어 정보 추출
    scores = [doc.get("score", 0.0) for doc in retrieved_documents]
    max_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    # 매칭된 문서 조각 요약 정보
    matched_sources = [
        {
            "source_name": doc.get("metadata", {}).get("source_name", "Unknown"),
            "score": round(doc.get("score", 0.0), 4),
            "text_preview": doc.get("text", "")[:50] + "..."  # 분석을 위해 앞부분 50자만 기록
        }
        for doc in retrieved_documents
    ]
    
    # 새 로그 객체 생성
    from datetime import datetime
    new_log = {
        "timestamp": datetime.now().isoformat(),
        "user_id": request.user_id,
        "project_id": request.project_id,
        "question": request.question,
        "answer_mode": request.answer_mode,
        "max_score": round(max_score, 4),
        "avg_score": round(avg_score, 4),
        "matched_sources": matched_sources
    }
    
    logs.append(new_log)
    
    # 최근 200개 로그만 유지하고 저장
    with LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(logs[-200:], f, ensure_ascii=False, indent=2)
