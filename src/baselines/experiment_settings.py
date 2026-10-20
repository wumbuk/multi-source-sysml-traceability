"""Single public configuration source for all three baselines."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SETTINGS = json.loads((ROOT / 'experiment_config.json').read_text(encoding='utf-8'))
for method in ('embedding', 'supervised'):
    model_path = Path(SETTINGS[method]['model_path'])
    if not model_path.is_absolute():
        SETTINGS[method]['model_path'] = str(ROOT / model_path)
