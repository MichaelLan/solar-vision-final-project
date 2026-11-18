import json
import pprint
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "artifacts" / "model.json"
target = ROOT / "backend" / "src" / "infra" / "model_payload.py"
model = json.loads(source.read_text(encoding="utf-8"))
target.write_text("MODEL = " + pprint.pformat(model, width=100, sort_dicts=False) + "\n", encoding="utf-8")
print(f"Packaged {source} into {target} ({target.stat().st_size} bytes)")
