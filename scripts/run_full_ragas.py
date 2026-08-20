from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragas_eval import load_cases, collect_rag_samples, build_ragas_metrics, run_ragas_evaluation

cases = load_cases(Path('d:/개발/AI-Server/evaluation/ragas_dataset.project101.json'))
print('loaded cases', len(cases))
collected = collect_rag_samples(cases, default_user_id=1)
print('collected samples', len(collected))
metrics = build_ragas_metrics()
print('built metrics:', list(metrics.keys()))
evaluated = run_ragas_evaluation(collected, metrics=metrics)
print('evaluated samples', len(evaluated))

output = {
    'generatedAt': None,
    'dataset': str(Path('d:/개발/AI-Server/evaluation/ragas_dataset.project101.json').resolve()),
    'models': {},
    'summary': None,
    'samples': evaluated,
}
from datetime import datetime, timezone
from statistics import mean
output['generatedAt'] = datetime.now(timezone.utc).isoformat()
output['summary'] = {
    'sampleCount': len(evaluated),
    'successfulSampleCount': sum(1 for s in evaluated if not s.get('errors')),
    'metrics': {},
    'averageResponseTimeMs': round(mean(s.get('responseTimeMs',0) for s in evaluated),2) if evaluated else None,
}
outpath = Path('d:/개발/AI-Server/data/ragas_results/ragas-full-run.json')
outpath.parent.mkdir(parents=True, exist_ok=True)
outpath.write_text(__import__('json').dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
print('wrote', outpath)
