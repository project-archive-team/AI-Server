import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
script = ROOT / 'scripts' / 'run_ragas_user1.py'
if not script.exists():
    print('runner error: script not found', script)
    raise SystemExit(1)

sys.argv = [str(script), str(ROOT / 'evaluation' / 'ragas_dataset.project101.json'), str(ROOT / 'data' / 'ragas_results' / 'ragas-full-101.json')]
try:
    runpy.run_path(str(script), run_name='__main__')
except Exception as e:
    import traceback
    traceback.print_exc()
    raise
