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
- `POST /career/star`: 지원 직무와 문항에 맞춘 구조화된 STAR 자기소개서 답변
- `POST /career/interview-questions`: 프로젝트 기반 구조화된 실전 면접 질문 카드
- `POST /portfolio/report`: 아카이브 자료 기반 구조화된 개발자 포트폴리오 리포트
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

## 취업도구 계약

```json
POST /career/star
{
  "projectId": 1,
  "jobRole": "백엔드 / AI 엔지니어",
  "question": "기술적 어려움을 극복하고 성과를 낸 경험을 서술해 주세요."
}
```

응답에는 `star.situation`, `star.task`, `star.action`, `star.result`,
`finalAnswer`, `missingEvidence`, `citations`가 포함됩니다.

```json
POST /career/interview-questions
{
  "projectId": 1,
  "jobRole": "백엔드 / AI 엔지니어",
  "questionCount": 3
}
```

각 질문은 `category`, `likelihood`, `modelAnswer`, `checkpoints`,
`followUps`, `citations`를 포함합니다. `questionCount`는 1~5입니다.

## 포트폴리오 리포트 계약

```json
POST /portfolio/report
{
  "projectId": 1,
  "projectName": "AI 기반 프로젝트 검색 및 아카이빙 플랫폼",
  "period": "2025.09 ~ 2025.12",
  "teamSize": "5명",
  "role": "백엔드 리드 및 AI 파이프라인"
}
```

`projectName`, `period`, `teamSize`, `role`은 선택 입력이며 백엔드에 저장된
프로젝트 메타데이터가 있으면 함께 전달하는 것을 권장합니다. 응답은 다음 화면 영역과
일대일로 대응합니다.

- `oneLineSummary`, `executiveSummary`: 프로젝트 한 줄 요약
- `techStack`, `systemArchitecture`, `dataPipeline`: 기술 스택 및 아키텍처
- `contributions`: 핵심 역할 및 기여
- `troubleshooting`: Situation / Action / Result 문제 해결 카드
- `retrospective`: 기술 성장 / 협업 인사이트 / 향후 개선점
- `missingEvidence`, `citations`: 부족한 근거와 원본 산출물 인용

코드·문서 원문 뷰어와 커밋·회의록 목록은 AI 생성 결과가 아니라 백엔드의 원본
산출물 API를 사용합니다.
