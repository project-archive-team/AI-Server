import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# 현재 config.py 파일이 있는 위치를 기준으로 .env 파일의 절대 경로를 계산합니다.
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        ".env 파일에서 GEMINI_API_KEY를 찾을 수 없습니다. "
        "파일 위치나 변수명을 다시 한 번 확인해 주세요."
    )

# Gemini 클라이언트 초기화
client = genai.Client(api_key=API_KEY)

# 모델 설정
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

# 벡터 DB 저장 경로
STORE_PATH = Path("data/vector_store.json")

# 팀 백엔드(Spring) 주소. 프론트엔드는 FastAPI의 /backend 프록시를 통해 호출합니다.
BACKEND_URL = os.getenv("BACKEND_URL", "http://13.125.136.195")
