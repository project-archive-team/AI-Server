import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import SimpleVectorStore, create_embedding
from config import STORE_PATH

print('STORE_PATH resolved as', STORE_PATH)
store = SimpleVectorStore()
print('Loaded documents:', len(store.documents))
count=0
for doc in store.documents:
    md = doc.get('metadata', {})
    if md.get('project_id') == 101:
        count += 1
        emb = doc.get('embedding')
        print('doc id', doc.get('id')[:8], 'user_id', md.get('user_id'), 'project_id', md.get('project_id'), 'source', md.get('source_name'), 'embedding_len', len(emb) if emb else None)
        if count>=10:
            break
print('project_id=101 count=', count)

# create embedding dim
try:
    emb = create_embedding('이 프로젝트의 RAG 파이프라인에서 문서 청킹과 임베딩 전략은 어떻게 설계되었고, 그 이유는 무엇인가요?')
    print('created embedding len', len(emb))
except Exception as e:
    print('embedding creation error', e)
    raise
