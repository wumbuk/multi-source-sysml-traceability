"""Public model settings; credentials are read from .env or the environment."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment_settings import SETTINGS

for env_path in (ROOT / '.env',):
    if env_path.is_file():
        for line in env_path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if key.strip().startswith(('LLM_', 'EXPERIMENT_')):
                    os.environ.setdefault(key.strip(), value.strip().strip('\"\''))

options = SETTINGS['llm']
base_url = os.getenv('LLM_BASE_URL', options['base_url'])
api_key = os.getenv('LLM_API_KEY', 'EMPTY')
model_name = os.getenv('LLM_MODEL', options['model'])
DEFAULT_TEST_PATH = os.getenv('EXPERIMENT_INPUT', str(ROOT / 'Data' / SETTINGS['dataset'] / 'Judge.csv'))
DEFAULT_OUTPUT_DIR = os.getenv('EXPERIMENT_OUTPUT_DIR', str(ROOT / 'results' / 'Llm_baseline' / model_name.rsplit('/', 1)[-1] / SETTINGS['dataset']))
LLM_PREDICTIONS_FILENAME = os.getenv('LLM_PREDICTIONS_FILENAME', 'llm_predictions.json')


def request_extra_body(model=None):
    """Null preserves the historical main experiment's server defaults."""
    thinking = options.get('thinking')
    selected_model = model or model_name
    extra = {}
    if thinking is not None:
        if not isinstance(thinking, bool):
            raise ValueError('llm.thinking must be null, true or false')
        if 'llama' in selected_model.lower():
            pass  # LLaMA-3.1-Instruct has no native Thinking switch.
        elif selected_model.lower().startswith('deepseek'):
            extra['thinking'] = {'type': 'enabled' if thinking else 'disabled'}
        else:
            extra['chat_template_kwargs'] = {'enable_thinking': thinking}
    penalty = options.get('repetition_penalty')
    if penalty is not None:
        extra['repetition_penalty'] = penalty
    return extra
