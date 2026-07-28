# 서버 테스트

GitHub Actions의 `.github/workflows/ci-cd.yml`은 GitHub가 제공하는 Ubuntu 서버에서
다음 검증을 수행합니다.

- Pull request 생성 및 갱신
- `main` 브랜치 push
- Actions 화면에서 수동 실행

검증 항목:

1. EC2 운영 환경과 동일한 Python 3.9 설치
2. `requirements-dev.txt` 의존성 설치
3. 전체 Python 소스 컴파일
4. pytest 실행

실제 Gemini API 키는 테스트 서버에 전달하지 않습니다. 애플리케이션 import에 필요한
자리표시자 키만 해당 작업 동안 사용합니다.

## EC2 배포

`main` 테스트가 통과하면 Amazon Linux EC2에 직접 배포하고 systemd 서비스를
재시작한 다음 `/health`를 확인합니다. Docker는 사용하지 않습니다.

GitHub 저장소의 **Settings → Secrets and variables → Actions**에 다음 Repository
Secrets를 등록해야 합니다.

- `EC2_HOST`: `13.125.136.195`
- `EC2_USER`: `ec2-user`
- `EC2_SSH_KEY`: `pojang.pem` 파일 내용 전체
- `GEMINI_API_KEY`: Gemini API 키
- `BACKEND_URL`: Spring 백엔드 주소

백엔드도 같은 EC2에서 실행되고 있다면 `BACKEND_URL`에는 백엔드가 실제로 수신하는
로컬 주소와 포트를 사용합니다. 예: `http://127.0.0.1:8080`.

백엔드의 `AI_BASE_URL`은 같은 서버에서 접근할 경우 `http://127.0.0.1:8000`으로
설정합니다. PostgreSQL, JWT, OAuth 관련 값은 백엔드 저장소에만 등록합니다.

로컬에서 같은 테스트를 실행하려면:

```powershell
$env:GEMINI_API_KEY = "test-key"
.\.venv\Scripts\python.exe -m pytest -q
```
