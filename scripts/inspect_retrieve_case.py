import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragas_eval import load_cases
from services import retrieve_project_context
from schemas import ChatRequest

cases = load_cases(Path('d:/개발/AI-Server/evaluation/ragas_dataset.project101.json'))
case = cases[0]
request = ChatRequest(user_id=1, project_id=case.projectId, question=case.question, answer_mode=case.answerMode, top_k=5)
results = retrieve_project_context(request)
print('retrieved count', len(results))
for r in results:
    print('id', r.get('id'), 'score', r.get('score'))
