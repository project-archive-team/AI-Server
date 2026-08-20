from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import client, GEMINI_CHAT_MODEL
import json

collected_path = Path('d:/개발/AI-Server/data/ragas_results/collect-only-user1-nollm.json')
collected = json.loads(collected_path.read_text(encoding='utf-8'))

samples = collected[:3]

prompt_parts = [
    "You are an evaluator. For each sample, rate the following metrics between 0 and 1 (float): faithfulness, answer_relevancy, context_precision, context_recall.\nReturn a JSON array of objects with keys: question, faithfulness, answer_relevancy, context_precision, context_recall, reason_short.\nOnly return valid JSON.",
]

for s in samples:
    prompt_parts.append("---")
    prompt_parts.append(f"Question: {s['user_input']}")
    prompt_parts.append(f"Reference: {s.get('reference','')}")
    prompt_parts.append(f"Retrieved Contexts: {s.get('retrieved_contexts',[]) }")
    prompt_parts.append(f"Response: {s.get('response','')}")

prompt = "\n".join(prompt_parts)

resp = client.models.generate_content(
    model=GEMINI_CHAT_MODEL,
    contents=prompt,
    config={
        'temperature': 0.0,
        'max_output_tokens': 800,
    }
)

out_text = resp.text
out_path = Path('d:/개발/AI-Server/data/ragas_results/llm-batch-judge.json')
out_path.write_text(json.dumps({'prompt': prompt, 'response': out_text}, ensure_ascii=False, indent=2), encoding='utf-8')
print('wrote', out_path)
