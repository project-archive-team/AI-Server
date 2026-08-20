from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragas_eval import load_cases
from services import retrieve_project_context
from schemas import ChatRequest
import json
import re

CASES_PATH = Path('d:/개발/AI-Server/evaluation/ragas_dataset.project101.json')
OUT_PATH = Path('d:/개발/AI-Server/data/ragas_results/heuristic-judge.json')

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def token_set(text: str):
    return set(m.group(0).lower() for m in WORD_RE.finditer(text or ""))


def compute_overlap(reference: str, retrieved_texts: list[str]) -> float:
    if not reference:
        return 0.0
    ref_tokens = token_set(reference)
    if not ref_tokens:
        return 0.0
    retrieved_tokens = set()
    for t in retrieved_texts:
        retrieved_tokens.update(token_set(t))
    return len(ref_tokens & retrieved_tokens) / len(ref_tokens)


cases = load_cases(CASES_PATH)
results = []
for case in cases:
    req = ChatRequest(user_id=1, project_id=case.projectId, question=case.question, answer_mode=case.answerMode, top_k=case.topK)
    docs = retrieve_project_context(req)
    retrieved_texts = [d.get('text','') for d in docs]
    scores = [d.get('score', 0.0) for d in docs]
    max_score = max(scores) if scores else 0.0
    mean_score = sum(scores)/len(scores) if scores else 0.0
    count = len(docs)
    overlap = compute_overlap(case.reference, retrieved_texts)
    # normalize max_score into [0,1] assuming typical cosine ~[-1,1], clamp
    norm_max = max(0.0, min(1.0, (max_score + 1) / 2))
    # combined heuristic: weights tuned for recall/precision proxy
    combined = 0.5 * norm_max + 0.3 * overlap + 0.2 * min(1.0, count / (case.topK or 5))
    results.append({
        'projectId': case.projectId,
        'answerMode': case.answerMode,
        'question': case.question,
        'retrieved_count': count,
        'max_score': max_score,
        'mean_score': mean_score,
        'reference_overlap': overlap,
        'heuristic_score': round(combined, 4),
        'retrieved_texts': retrieved_texts[:5],
    })

summary = {
    'sampleCount': len(results),
    'avg_heuristic_score': round(sum(r['heuristic_score'] for r in results)/len(results),4) if results else None,
    'avg_retrieved_count': round(sum(r['retrieved_count'] for r in results)/len(results),2) if results else None,
}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps({'generatedAt': __import__('datetime').datetime.utcnow().isoformat() + 'Z', 'summary': summary, 'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
print('wrote', OUT_PATH)
