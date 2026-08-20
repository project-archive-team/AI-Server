from pathlib import Path
import json
import sys

# Ensure project root (AI-Server) is on sys.path so `config` imports correctly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from config import client, GEMINI_EMBEDDING_MODEL
except Exception as e:
    print(json.dumps({"status": "error", "stage": "config_load", "error": str(e)}))
    raise SystemExit(2)

try:
    resp = client.models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents="test")
    print(json.dumps({"status": "ok", "detail": "embed_call_succeeded"}))
except Exception as e:
    print(json.dumps({"status": "error", "stage": "embed_call", "error": str(e)}))
    raise SystemExit(3)
