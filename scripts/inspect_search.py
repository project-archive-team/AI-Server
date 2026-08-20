import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import SimpleVectorStore, create_embedding
from config import STORE_PATH

print('STORE_PATH resolved as', STORE_PATH)
store = SimpleVectorStore()
print('Loaded documents:', len(store.documents))

q = '프로젝트에서 사용된 주요 기술 스택과 설계 결정을 요약해줘.'
print('Test query:', q)
emb = create_embedding(q)
print('created embedding len', len(emb))

results = store.search(emb, top_k=5, user_id=1, project_id=101)
print('search returned', len(results), 'items')
for i, r in enumerate(results, 1):
    try:
        print(f'Result {i}:', r)
    except Exception:
        print(f'Result {i}: repr ->', repr(r))
