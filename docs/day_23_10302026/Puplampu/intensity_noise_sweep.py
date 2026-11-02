"""Compare repeated measurements at eight energies, duration fixed to one."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import requests
OUT=Path(__file__).resolve().parent
FILE=OUT/'challenge1_intensity_noise_sweep.json'
BASE='https://athena-spectral-api.onrender.com'
def api(method,path,payload=None,token=None):
    r=requests.request(method,BASE+path,json=payload,headers={'X-Session-ID':token} if token else {},timeout=120)
    r.raise_for_status();return r.json()
if FILE.exists():raise RuntimeError('Existing experiment found. Inspect before rerunning; no automatic retries.')
info=api('GET','/info')
energies=[0.,1.,2.5,3.8,5.,6.,7.3,9.]
assert min(energies)>=info['wavelength_range'][0] and max(energies)<=info['wavelength_range'][1]
state={'created_utc':datetime.now(timezone.utc).isoformat(),'energies':energies,'duration':1.,'repeats':30,'sessions':[],'observations':[]}
def save():FILE.write_text(json.dumps(state,indent=2))
save();rng=np.random.default_rng(2304);remaining=0.;token=None
for block in range(3):
    schedule=np.repeat(energies,10);rng.shuffle(schedule)
    for energy in schedule:
        if token is None or remaining<1-1e-9:
            session=api('POST','/session',{'task':'single-spectrum'})
            token=session['session_id'];remaining=float(session['time_budget'])
            state['sessions'].append({'session_id':token,'time_budget':remaining});save()
        result=api('POST','/pixel',{'wavelength':float(energy),'time':1.},token)
        state['observations'].append({'block':block+1,'session_index':len(state['sessions'])-1,'energy':float(energy),'result':result})
        remaining=float(result.get('time_remaining',remaining-1));save()
        if len(state['observations'])%30==0:print(f'{len(state["observations"])}/240 observations saved',flush=True)
print('Intensity/noise sweep complete. No submissions.',flush=True)
