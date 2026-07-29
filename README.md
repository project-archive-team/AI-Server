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
- `POST /index/delete`: 삭제된 artifact ID 목록의 벡터 청크 제거
- `DELETE /index/projects/{project_id}`: 프로젝트 전체 벡터 청크 제거
- `GET /health`: 상태 확인

Spring 백엔드의 `AI_BASE_URL`에는 이 서버 주소를 설정합니다.

소스나 프로젝트를 백엔드에서 삭제할 때 AI 색인에도 삭제를 전달해야 합니다.

```json
POST /index/delete
{
  "projectId": 1,
  "artifactIds": [10, 11]
}
```

프로젝트 삭제 시에는 `DELETE /index/projects/1`을 호출합니다. 삭제 호출은 멱등적이며
이미 없는 청크는 오류 없이 `deleted: 0`을 반환합니다.
