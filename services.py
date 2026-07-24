import json
import os
from pathlib import Path
from uuid import uuid4
import numpy as np
from typing import Any
from pydantic import BaseModel
from google.genai import types

# 프로젝트 내의 다른 모듈 불러오기
from config import client, GEMINI_EMBEDDING_MODEL, GEMINI_CHAT_MODEL, STORE_PATH
from schemas import DocumentRequest, ChatRequest


# 구조화된 출력을 위한 스키마 정의
class InitialQuestionsSchema(BaseModel):
    recommended_questions: list[str]


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

class SimpleVectorStore:
    def __init__(self, path=STORE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self.documents, file, ensure_ascii=False, indent=2)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return
        self.documents.extend(documents)
        self._save()

    def delete_source(self, user_id: int, project_id: int, source_name: str) -> int:
        before_count = len(self.documents)
        self.documents = [
            doc for doc in self.documents
            if not (
                doc.get("metadata", {}).get("user_id") == user_id
                and doc.get("metadata", {}).get("project_id") == project_id
                and doc.get("metadata", {}).get("source_name") == source_name
            )
        ]
        deleted_count = before_count - len(self.documents)
        if deleted_count > 0:
            self._save()
        return deleted_count

    def search(self, query_embedding: list[float], user_id: int, project_id: int | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        scored_documents: list[dict[str, Any]] = []
        for document in self.documents:
            metadata = document.get("metadata", {})
            if metadata.get("user_id") != user_id:
                continue
            if project_id is not None and metadata.get("project_id") != project_id:
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


def retrieve_project_context(request: ChatRequest) -> list[dict[str, Any]]:
    query_embedding = create_embedding(request.question)
    vector_store = SimpleVectorStore()
    return vector_store.search(
        query_embedding=query_embedding,
        user_id=request.user_id,
        project_id=request.project_id,
        top_k=request.top_k,
    )


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

## 예상 꼬리 질문
이 답변을 들은 면접관이 추가로 물어볼 만한 날카로운 질문을 3개 작성합니다.
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