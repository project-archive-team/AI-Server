# Project Archive AI Server

Spring 백엔드가 수집·파싱·청킹한 프로젝트 자료를 임베딩하고, RAG 답변을 생성하는 FastAPI 서버입니다.

## 실행

```bash
python -m venv .venv
pip install -r requirements.txt
cp .env.example .env
uvicorn ai_app:app --host 0.0.0.0 --port 8000
```

`.env`의 `GEMINI_API_KEY`를 실제 키로 교체해야 합니다.

## Backend contract

- `POST /index`: 청크 임베딩 및 프로젝트 색인
- `POST /chat`: 프로젝트 RAG Q&A
- `POST /summary`: 특정 시점 이후 활동 요약
- `POST /interview`: 프로젝트 기반 면접 답변
- `GET /health`: 상태 확인

Spring 백엔드의 `AI_BASE_URL`에는 이 서버 주소를 설정합니다.
