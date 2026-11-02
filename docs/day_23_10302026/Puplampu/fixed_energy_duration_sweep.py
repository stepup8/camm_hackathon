"""Repeated measurements at E=5; no predictions or submissions."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import requests
OUT=Path(__file__).resolve().parent
FILE=OUT/'challenge1_fixed_energy_duration_sweep.json'
BASE='https://athena-spectral-api.onrender.com'
def api(method,path,payload=None,token=None):
    r=requests.request(method,BASE+path,json=payload,headers={'X-Session-ID':token} if token else {},timeout=120)
    r.raise_for_status()
    return r.json()
if FILE.exists():raise RuntimeError('Existing results found; inspect before rerunning. No automatic retries.')
info=api('GET','/info')
assert info['wavelength_range'][0]<=5<=info['wavelength_range'][1]
state={'created_utc':datetime.now(timezone.utc).isoformat(),'energy':5.,'durations':[.25,.5,1.,2.,3.],'repeats_per_duration':30,'sessions':[],'observations':[]}
def save():FILE.write_text(json.dumps(state,indent=2))
save()
# Three randomized blocks; each block includes ten repeats of every duration.
# Sessions share one composition. Record means by session to inspect offsets.
rng=np.random.default_rng(2301)
remaining=0.;token=None
for block in range(3):
    schedule=np.repeat(state['durations'],10)
    rng.shuffle(schedule)
    for duration in schedule:
        duration=float(duration)
        if token is None or remaining < duration-1e-9:
            episode=api('POST','/session',{'task':'single-spectrum'})
            token=episode['session_id'];remaining=float(episode['time_budget'])
            state['sessions'].append({'session_id':token,'time_budget':remaining});save()
        result=api('POST','/pixel',{'wavelength':5.,'time':duration},token)
        record={'block':block+1,'session_index':len(state['sessions'])-1,'requested_duration':duration,'result':result}
        state['observations'].append(record)
        remaining=float(result.get('time_remaining',remaining-duration))
        save()
        if len(state['observations'])%10==0:print(f'{len(state["observations"])}/150 repeats saved; sessions={len(state["sessions"])}',flush=True)
print('Acquisition complete. No submissions made.',flush=True)
