from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
from ragas_eval import load_cases, run_ragas_evaluation, build_ragas_metrics

# Load collected samples (we have collect-only-user1-nollm.json)
collected_path = Path('d:/개발/AI-Server/data/ragas_results/collect-only-user1-nollm.json')
collected = json.loads(collected_path.read_text(encoding='utf-8'))

# choose small sample (first 3)
sample_count = 3
samples = collected[:sample_count]

metrics = build_ragas_metrics()
print('Built metrics:', list(metrics.keys()))

# run judge
evaluated = run_ragas_evaluation(samples, metrics=metrics)

out = {
    'generatedAt': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
    'source': str(collected_path),
    'sampleCount': len(evaluated),
    'evaluated': evaluated,
}
out_path = Path('d:/개발/AI-Server/data/ragas_results/ragas-sample-judge.json')
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('Wrote', out_path)
