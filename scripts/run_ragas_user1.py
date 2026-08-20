import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure AI-Server on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragas_eval import load_cases, run_ragas_evaluation, summarize_results
from services import retrieve_project_context, generate_answer
from schemas import ChatRequest


def main(dataset_path: str, output_path: str):
    dataset = Path(dataset_path)
    out = Path(output_path)
    cases = load_cases(dataset)
    samples = []
    from time import perf_counter
    clock = perf_counter
    for case in cases:
        request = ChatRequest(
            user_id=1,
            project_id=case.projectId,
            question=case.question,
            answer_mode=case.answerMode,
            top_k=case.topK,
        )
        started = clock()
        docs = retrieve_project_context(request)
        response = generate_answer(request, docs)
        rt = round((clock() - started) * 1000, 2)
        samples.append(
            {
                "projectId": case.projectId,
                "answerMode": case.answerMode,
                "user_input": case.question,
                "retrieved_contexts": [d.get("text", "") for d in docs],
                "response": response,
                "responseTimeMs": rt,
                "reference": case.reference,
            }
        )

    # run RAGAS evaluation
    try:
        evaluated = run_ragas_evaluation(samples)
    except Exception as e:
        # if evaluation fails, still write collected samples
        evaluated = samples

    summary = summarize_results(evaluated)
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "models": {},
        "summary": summary,
        "samples": evaluated,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote results to {out}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: run_ragas_user1.py <dataset.json> <output.json>")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
