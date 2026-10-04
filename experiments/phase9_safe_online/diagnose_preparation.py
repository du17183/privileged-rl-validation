import json
import h5py
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'datasets/phase9_safe_online/anchor_seed0.h5'
with h5py.File(path,'r') as f:
    count=sum(g.attrs['success']==1 for g in f.values())
    result=dict(original_expression_type=type(count).__name__)
    try:
        json.dumps(dict(successes=count))
    except TypeError as error:
        result['reproduced_serialization_error']=str(error)
    result['integer_cast_works']=json.loads(json.dumps(dict(successes=int(count))))['successes']==int(count)
(ROOT/'results/phase9_safe_online/prepare_serialization_diagnosis.json').write_text(json.dumps(result,indent=2))
print(result)
