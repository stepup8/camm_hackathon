"""Equal-budget uniform sampling sweep. Uses only documented participant API."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import requests

OUT=Path(__file__).resolve().parent
FILE=OUT/'challenge1_time_sweep.json'
BASE='https://athena-spectral-api.onrender.com'
def api(method,path,payload=None,token=None):
    response=requests.request(method,BASE+path,json=payload,
        headers={'X-Session-ID':token} if token else {},timeout=120)
    response.raise_for_status()
    return response.json()
if FILE.exists():
    raise RuntimeError('Existing experiment found. Inspect before rerunning; no automatic retries.')
info=api('GET','/info')
energy=np.linspace(*info['wavelength_range'],info['channels'])
state={'created_utc':datetime.now(timezone.utc).isoformat(),
       'method':'Uniform coordinates, equal duration per observation, linear interpolation',
       'counts':[10,15,30,60,120], 'repeats':3, 'energy':energy.tolist(),'episodes':[]}
def save(): FILE.write_text(json.dumps(state,indent=2))
save()
rng=np.random.default_rng(23)
for repeat in range(3):
    # Randomize setting order in each block to reduce confounding with time.
    for count in rng.permutation(state['counts']):
        count=int(count)
        episode=api('POST','/session',{'task':'single-spectrum'})
        duration=float(episode['time_budget'])/count
        record={'repeat':repeat+1,'count':count,'duration':duration,
                'time_budget':float(episode['time_budget']),
                'session_id':episode['session_id'],'observations':[]}
        state['episodes'].append(record);save()
        print(f'Start: repeat {repeat+1}/3, N={count}, duration={duration:g}',flush=True)
        wavelengths=np.linspace(energy.min(),energy.max(),count)
        for i,wavelength in enumerate(wavelengths):
            result=api('POST','/pixel',{'wavelength':float(wavelength),'time':duration},record['session_id'])
            record['observations'].append(result);save()
            if (i+1)%30==0: print(f'  {i+1}/{count} measurements saved',flush=True)
        x=np.array([r['wavelength'] for r in record['observations']])
        y=np.array([r['intensity'] for r in record['observations']])
        order=np.argsort(x)
        prediction=np.interp(energy,x[order],y[order])
        assert np.isfinite(prediction).all()
        record['prediction']=prediction.tolist()
        record['submission_attempted']=True;save()
        record['score']=api('POST','/submit',{'energy':energy.tolist(),
            'predicted_intensity':prediction.tolist()},record['session_id']);save()
        print('Score:',json.dumps(record['score']),flush=True)
print('Sweep complete.',flush=True)
