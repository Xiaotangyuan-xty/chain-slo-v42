"""Domain routing only in v48; original DR fit and pure forecast stay intact."""
from pathlib import Path
import json
from chain_forecast import StageForecaster

BANK = None


def tot_model(model_key):
    global BANK
    if BANK is None:
        path = Path(__file__).resolve().parents[1] / 'assets/TOT_CALIBRATION.json'
        data = json.loads(path.read_text())
        BANK = {key: StageForecaster(value['model']) for key, value in data['models'].items()}
    return BANK.get(model_key)
