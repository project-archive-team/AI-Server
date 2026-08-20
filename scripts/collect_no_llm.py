from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragas_eval import load_cases, collect_rag_samples

# simple answer generator to avoid LLM calls during collection
def stub_generator(request, docs):
    return "[COLLECT-ONLY]"

cases = load_cases(Path('d:/개발/AI-Server/evaluation/ragas_dataset.project101.json'))
collected = collect_rag_samples(cases, answer_generator=stub_generator, default_user_id=1)
out = Path('d:/개발/AI-Server/data/ragas_results/collect-only-user1-nollm.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(__import__('json').dumps(collected, ensure_ascii=False, indent=2), encoding='utf-8')
print('wrote', out)
