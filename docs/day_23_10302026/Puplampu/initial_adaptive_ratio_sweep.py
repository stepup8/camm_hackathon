"""Compare initial/adaptive splits with the original GP and placement rule."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json,warnings
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import requests
OUT=Path(__file__).resolve().parent
FILE=OUT/'challenge1_ratio_sweep.json'
BASE='https://athena-spectral-api.onrender.com'
def api(method,path,payload=None,token=None):
    response=requests.request(method,BASE+path,json=payload,headers={'X-Session-ID':token} if token else {},timeout=120)
    response.raise_for_status()
    return response.json()
if FILE.exists():raise RuntimeError('Existing experiment found; inspect before rerunning. No automatic retries.')
# Reuse the exact original model and acquisition functions, without running live cells.
namespace={}
notebook=json.loads((OUT/'challenge1_adaptive.ipynb').read_text())
for cell in notebook['cells']:
    source=''.join(cell['source'])
    if cell['cell_type']=='code' and ('import json' in source or 'def fit_model' in source):
        exec(compile(source,'original_notebook_helpers','exec'),namespace)
info=api('GET','/info')
energy=np.linspace(*info['wavelength_range'],info['channels'])
state={'created_utc':datetime.now(timezone.utc).isoformat(),'initial_counts':[5,10,15,20],
       'total_measurements':30,'repeats':3,'energy':energy.tolist(),
       'method':'Original GP; integrated variance reduction; every fifth adaptive observation uses coverage. Equal durations.',
       'episodes':[]}
def save():FILE.write_text(json.dumps(state,indent=2))
save()
rng=np.random.default_rng(2303)
for repeat in range(3):
    for initial_count in rng.permutation(state['initial_counts']):
        initial_count=int(initial_count)
        namespace['N_INITIAL']=initial_count
        episode=api('POST','/session',{'task':'single-spectrum'})
        token=episode['session_id'];duration=float(episode['time_budget'])/30
        record={'repeat':repeat+1,'initial_count':initial_count,'adaptive_count':30-initial_count,
                'session_id':token,'duration':duration,'time_budget':float(episode['time_budget']),
                'observations':[],'model_warnings':[]}
        state['episodes'].append(record);save()
        initial_indices=np.rint(np.linspace(0,len(energy)-1,initial_count)).astype(int)
        print(f'Start repeat {repeat+1}/3: {initial_count} initial + {30-initial_count} adaptive',flush=True)
        for i in range(30):
            if i<initial_count:
                index,rule=int(initial_indices[i]),'initial coverage'
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    index,rule=namespace['choose_next'](energy,record['observations'])
                record['model_warnings'].append({'after_measurements':i,'warnings':[str(w.message) for w in caught]})
            result=api('POST','/pixel',{'wavelength':float(energy[index]),'time':duration},token)
            result.update(requested_wavelength=float(energy[index]),requested_time=duration,placement_rule=rule)
            record['observations'].append(result);save()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            gp,prediction,_,_=namespace['fit_model'](energy,record['observations'])
        assert prediction.shape==energy.shape and np.isfinite(prediction).all()
        record['kernel']=str(gp.kernel_)
        record['final_fit_warnings']=[str(w.message) for w in caught]
        record['prediction']=prediction.tolist()
        record['submission_attempted']=True;save()
        record['score']=api('POST','/submit',{'energy':energy.tolist(),'predicted_intensity':prediction.tolist()},token)
        save()
        print('Score:',json.dumps(record['score']),flush=True)
print('Ratio sweep complete.',flush=True)
