import json
from pathlib import Path

SRC = Path('d:/개발/AI-Server/evaluation/ragas_dataset.generated.mapped.json')
DST = Path('d:/개발/AI-Server/evaluation/ragas_dataset.project101.json')

items = json.loads(SRC.read_text(encoding='utf-8'))
for it in items:
    it['projectId'] = 101
DST.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Wrote {DST} ({len(items)} items)')
