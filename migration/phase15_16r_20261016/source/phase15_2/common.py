import json
from pathlib import Path
from phase15_1.common import ROOT,AUTH,frozen,sha256,save_json
RESULTS=ROOT/'phase15_2/results'
LOGS=ROOT/'logs/phase15_2'
def isolated(path):
    path=Path(path).resolve()
    if not path.is_relative_to(RESULTS.resolve()):raise RuntimeError('Output escaped Phase15.2')
    return path
def redirect_calibration():
    import phase15_1.severe_reacquire.pregrasp_planner as old
    old.RESULTS=RESULTS
def read(path):return json.loads(Path(path).read_text())
