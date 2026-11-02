"""Validate duration scaling at three additional energies. No submissions."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import requests
OUT=Path(__file__).resolve().parent
FILE=OUT/'challenge1_multi_energy_duration_sweep.json'
BASE='https://athena-spectral-api.onrender.com'
def api(method,path,payload=None,token=None):
    r=requests.request(method,BASE+path,json=payload,headers={'X-Session-ID':token} if token else {},timeout=120)
    r.raise_for_status();return r.json()
if FILE.exists():raise RuntimeError('Existing experiment found. Inspect before rerunning; no automatic retries.')
info=api('GET','/info')
energies=[0.,6.,7.3];durations=[.25,.5,1.,2.,3.]
assert min(energies)>=info['wavelength_range'][0] and max(energies)<=info['wavelength_range'][1]
state={'created_utc':datetime.now(timezone.utc).isoformat(),'energies':energies,'durations':durations,'repeats':30,'sessions':[],'observations':[]}
def save():FILE.write_text(json.dumps(state,indent=2))
save();rng=np.random.default_rng(2306);remaining=0.;token=None
for block in range(3):
    schedule=[(e,t) for e in energies for t in durations for _ in range(10)]
    rng.shuffle(schedule)
    for energy,duration in schedule:
        if token is None or remaining<duration-1e-9:
            episode=api('POST','/session',{'task':'single-spectrum'})
            token=episode['session_id'];remaining=float(episode['time_budget'])
            state['sessions'].append({'session_id':token,'time_budget':remaining});save()
        result=api('POST','/pixel',{'wavelength':energy,'time':duration},token)
        state['observations'].append({'block':block+1,'session_index':len(state['sessions'])-1,'energy':energy,'duration':duration,'result':result})
        remaining=float(result.get('time_remaining',remaining-duration));save()
        if len(state['observations'])%30==0:print(f'{len(state["observations"])}/450 measurements saved; sessions={len(state["sessions"])}',flush=True)
print('Acquisition complete. No submissions.',flush=True)
