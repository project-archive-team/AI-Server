import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragas_eval import load_cases, collect_rag_samples

cases = load_cases(Path('d:/개발/AI-Server/evaluation/ragas_dataset.project101.json'))
samples = collect_rag_samples(cases, default_user_id=1)
print('samples loaded', len(samples))
for i, s in enumerate(samples[:5], 1):
    print(i, 'retrieved count:', len(s.get('retrieved_contexts', [])))
    print('response snippet:', s.get('response')[:80])
