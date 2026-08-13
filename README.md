# Project Archive AI Server

Spring 백엔드가 수집·파싱·청킹한 프로젝트 자료를 임베딩하고, 검색된 근거만으로
RAG 답변과 취업 지원 콘텐츠를 생성하는 FastAPI 서버입니다.

## 주요 기능

- 프로젝트별 `COMMIT`, `CODE`, `DOC`, `MEETING` 청크 색인
- Gemini 임베딩과 코사인 유사도 기반 문맥 검색
- 프로젝트 Q&A, 활동 요약, 프로젝트 기반 면접 답변 생성
- 지원 직무에 맞춘 구조화된 STAR 자기소개서 생성
- STAR 형식의 실전 면접 예상 질문·모범 답변·꼬리질문 생성
- 프로젝트 자료 기반 개발자 포트폴리오 리포트 생성
- artifact 또는 프로젝트 단위의 색인 교체·삭제
- RAGAS 4개 지표와 응답 시간 기반 배치 품질 평가

## RAG 구조

이 서버는 사전 학습된 Gemini 모델을 연결한 모듈식 RAG 파이프라인입니다.

```text
Spring 백엔드의 프로젝트 청크
  → Gemini Embedding API
  → data/vector_store.json에 프로젝트·artifact 메타데이터와 함께 저장

사용자 질문
  → 질문 임베딩
  → 프로젝트 필터 + 코사인 유사도 검색
  → 상위 8개 문맥
  → Gemini 생성 모델
  → 답변 + 원본 artifact 인용
```

새 자료가 들어오면 해당 청크를 다시 임베딩하여 색인을 교체합니다.
현재 벡터 저장소는 로컬 JSON 파일이므로 단일 서버 환경을 전제로 합니다.

## 실행 환경

- Python 3.9 이상
- Gemini API 키
- 기본 임베딩 모델: `gemini-embedding-001`
- 기본 생성 모델: `gemini-2.5-flash`

```bash
python -m venv .venv
pip install -r requirements.txt
cp .env.example .env
uvicorn ai_app:app --host 0.0.0.0 --port 8000
```

`.env` 설정:

| 변수 | 필수 | 기본값 | 용도 |
|---|---|---|---|
| `GEMINI_API_KEY` | 예 | 없음 | Gemini API 인증 |
| `GEMINI_EMBEDDING_MODEL` | 아니요 | `gemini-embedding-001` | 문서·질문 임베딩 |
| `GEMINI_CHAT_MODEL` | 아니요 | `gemini-2.5-flash` | 답변 및 구조화 콘텐츠 생성 |
| `BACKEND_URL` | 아니요 | `http://13.125.136.195` | Spring 백엔드 주소 |
| `RAGAS_EVALUATOR_MODEL` | 아니요 | 생성 모델과 동일 | RAGAS judge 모델 |
| `RAGAS_DO_NOT_TRACK` | 아니요 | `true` | RAGAS 텔레메트리 비활성화 |

상태 확인은 `GET /health`로 수행합니다. Spring 백엔드의 `AI_BASE_URL`에는 이
서버의 주소를 설정합니다.

## API 계약

| Method | Path | 설명 |
|---|---|---|
| `POST` | `/index` | 청크 임베딩 및 프로젝트 색인 |
| `POST` | `/index/delete` | artifact ID별 색인 삭제 |
| `DELETE` | `/index/projects/{project_id}` | 프로젝트 전체 색인 삭제 |
| `POST` | `/chat` | 프로젝트 기반 RAG Q&A |
| `POST` | `/summary` | 특정 시점 이후 활동 요약 |
| `POST` | `/interview` | 프로젝트 기반 단일 면접 답변 |
| `POST` | `/career/star` | 구조화된 STAR 자기소개서 답변 |
| `POST` | `/career/interview-questions` | 실전 면접 질문 카드 생성 |
| `POST` | `/portfolio/report` | 구조화된 포트폴리오 리포트 생성 |
| `GET` | `/health` | 서버 상태 확인 |

### 프로젝트 색인

Spring 백엔드가 파싱과 청킹을 마친 자료를 전달합니다.

```json
POST /index
{
  "projectId": 1,
  "chunks": [
    {
      "artifactId": 10,
      "type": "DOC",
      "title": "README.md",
      "path": "README.md",
      "url": "https://github.com/example/repository/blob/main/README.md",
      "seq": 0,
      "text": "프로젝트 문서의 청크 내용"
    }
  ]
}
```

같은 프로젝트와 artifact를 다시 색인하면 기존 청크를 중복 추가하지 않고 새 청크로
교체합니다. 소스나 프로젝트를 백엔드에서 삭제할 때 AI 색인에도 삭제 요청을 전달해야
합니다.

```json
POST /index/delete
{
  "projectId": 1,
  "artifactIds": [10, 11]
}
```

프로젝트 전체 삭제는 `DELETE /index/projects/1`을 호출합니다. 삭제 API는 멱등적이며
이미 존재하지 않는 청크는 오류 없이 `deleted: 0`을 반환합니다.

### 기본 RAG 답변

`/chat`과 `/interview`의 요청 형식은 같습니다.

```json
{
  "projectId": 1,
  "question": "캐시를 도입한 이유와 본인의 기여를 설명해 주세요."
}
```

응답은 `answer`와 검색 근거인 `citations`를 포함합니다. `/summary`는 `projectId`와
ISO 8601 형식의 `since`를 받아 해당 시점 이후의 자료만 검색합니다.

## 취업 지원 도구

모든 생성 결과는 검색된 프로젝트 자료에 근거합니다. 자료에서 확인할 수 없는 기술,
역할, 장애 원인, 성과 또는 수치는 만들지 않고 부족한 근거를 별도로 표시합니다.
답변은 지원 직무와 자료에서 확인되는 본인의 담당 역할(AI, 백엔드, 프론트엔드 등)이
직접 연결되는 경험을 우선 사용합니다. 프로젝트 전체 성과보다 본인이 맡은 책임,
판단, 행동과 기술적 기여를 중심으로 설명하며, 지원 직무와 기존 역할이 다른 경우에는
자료로 확인되는 전이 가능한 역량만 연결합니다.

### STAR 자기소개서

```json
POST /career/star
{
  "projectId": 1,
  "jobRole": "백엔드 / AI 엔지니어",
  "question": "기술적 어려움을 극복하고 성과를 낸 경험을 서술해 주세요."
}
```

응답에는 다음 필드가 포함됩니다.

- `star.situation`, `star.task`, `star.action`, `star.result`
- STAR 내용을 자연스럽게 연결한 `finalAnswer`
- 자료에서 확인할 수 없는 내용을 나타내는 `missingEvidence`
- 근거 artifact의 `citations`

### 실전 면접 예상 질문과 모범 답변

```json
POST /career/interview-questions
{
  "projectId": 1,
  "jobRole": "백엔드 / AI 엔지니어",
  "questionCount": 3
}
```

`questionCount`는 1~5이며 각 질문 카드는 `category`, `likelihood`, `question`,
`modelAnswer`, `checkpoints`, `followUps`, `citations`를 포함합니다. 꼬리질문은 항상
2개이며 각각 `question`과 근거 기반 `recommendedAnswer`를 제공합니다.

`modelAnswer`는 다음 원칙과 출력 형식을 따릅니다.

```text
S: 직무 역량을 보여주는 데 필요한 상황과 배경
T: 지원자 본인이 맡은 과제와 책임 (필요한 경우 팀 전체 목표 포함)
A: 지원자가 직접 수행한 판단, 행동, 기술적 선택과 기여
R: 프로젝트 자료에서 검증할 수 있는 성과와 변화
```

각 항목은 반드시 `S:`, `T:`, `A:`, `R:` 라벨로 시작하고 줄을 바꿔 출력합니다.
팀의 수행 내용과 지원자 개인의 기여를 구분하며, 단순 업무 수행을 성과처럼 포장하지
않습니다.

## 포트폴리오 리포트

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

`projectName`, `period`, `teamSize`, `role`은 선택 입력입니다. 백엔드에 저장된
프로젝트 메타데이터가 있으면 함께 전달하는 것을 권장합니다.

- `oneLineSummary`, `executiveSummary`: 프로젝트 핵심 요약
- `techStack`, `systemArchitecture`, `dataPipeline`: 기술 스택과 아키텍처
- `contributions`: 역할, 기여와 검증 가능한 성과
- `troubleshooting`: Situation / Action / Result 문제 해결 카드
- `retrospective`: 기술 성장, 협업 인사이트와 향후 개선점
- `missingEvidence`, `citations`: 부족한 근거와 원본 산출물 인용

코드·문서 원문 뷰어와 커밋·회의록 목록은 AI 생성 결과가 아니라 백엔드의 원본
산출물 API를 사용합니다.

## RAGAS 품질 평가

RAGAS 평가는 운영 API와 분리된 배치 CLI로 실행합니다. 공개 데이터셋 대신 실제 팀
프로젝트 자료를 기반으로 질문과 기준 답변의 초안을 만들고, 사람이 원문 근거·정답성·
모호성·중복 여부를 검토한 골든 데이터셋을 사용하는 방식입니다.

평가 의존성을 별도로 설치합니다.

```bash
pip install -r requirements-eval.txt
```

[`evaluation/ragas_dataset.example.json`](evaluation/ragas_dataset.example.json)을 복사해
실제 색인된 프로젝트 ID와 검수 완료된 질문·기준 답변으로 수정합니다.

```json
[
  {
    "projectId": 1,
    "question": "Redis 캐시를 도입한 이유는 무엇인가요?",
    "reference": "반복 조회 부하를 줄이기 위해 Redis 캐시를 도입했습니다.",
    "answerMode": "general",
    "topK": 8
  }
]
```

평가 실행:

```bash
python ragas_eval.py evaluation/ragas_dataset.example.json
```

평가 과정에서 실제 `retrieve_project_context`와 `generate_answer`를 호출하므로 대상
프로젝트가 먼저 색인되어 있어야 하며 Gemini API 사용량이 발생합니다.

| 항목 | 의미 |
|---|---|
| `faithfulness` | 답변이 검색 문맥에 근거하는지 평가 |
| `answer_relevancy` | 답변이 질문 의도에 맞는지 평가 |
| `context_precision` | 관련 근거가 검색 결과 상단에 배치되는지 평가 |
| `context_recall` | 기준 답변에 필요한 근거가 충분히 검색됐는지 평가 |
| `responseTimeMs` | 검색 시작부터 답변 생성 완료까지 걸린 시간 |

결과 JSON에는 질문별 검색 문맥, 실제 답변, 기준 답변, 4개 RAGAS 점수와 응답 시간이
저장됩니다. 요약에는 각 지표의 평균과 `averageResponseTimeMs`가 포함됩니다. 기본
저장 위치는 `data/ragas_results/ragas-<timestamp>.json`입니다.

평균 점수 기준을 적용하면 회귀 검증에도 사용할 수 있습니다.

```bash
python ragas_eval.py evaluation/ragas_dataset.example.json \
  --threshold faithfulness=0.8 \
  --threshold context_recall=0.7
```

RAGAS judge 호출 없이 검색 문맥, 실제 답변과 응답 시간만 수집하려면
`--collect-only`를 사용합니다. 출력 위치는 `--output <path>`로 지정할 수 있습니다.

## 테스트와 배포

로컬 테스트:

```powershell
$env:GEMINI_API_KEY = "test-key"
.\.venv\Scripts\python.exe -m pytest -q
```

GitHub Actions는 Python 3.9 환경에서 소스 컴파일과 pytest를 실행합니다. `main` 검증이
통과하면 Amazon Linux EC2에 배포하고 systemd 서비스를 재시작한 뒤 `/health`를
확인합니다. 배포 환경과 GitHub Secrets 설정은 [`CI.md`](CI.md)를 참고하세요.
