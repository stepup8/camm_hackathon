"""Organizer-scored uniform count/duration sweep, fixed Gaussian fitter."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import requests
from adaptive_peak_model import fit
OUT=Path(__file__).resolve().parent
FILE=OUT/'challenge1_peak_30_vs_60_scored.json'
BASE='https://athena-spectral-api.onrender.com'
def api(method,path,payload=None,token=None):
    r=requests.request(method,BASE+path,json=payload,headers={'X-Session-ID':token} if token else {},timeout=120)
    r.raise_for_status();return r.json()
if FILE.exists():raise RuntimeError('Existing experiment found; inspect before rerunning. No automatic retries.')
info=api('GET','/info');energy=np.linspace(*info['wavelength_range'],info['channels'])
state={'created_utc':datetime.now(timezone.utc).isoformat(),'counts':[30,60],'repeats_per_count':30,'energy':energy.tolist(),
'design':'Fresh focused confirmation; earlier screening episodes excluded from primary analysis','method':'Uniform locations, equal durations within each episode, total budget 30, unchanged original BIC-selected 1–4 Gaussian peaks plus linear background; calibrated-variance weighting.',
'episodes':[]}
def save():FILE.write_text(json.dumps(state,indent=2))
save();rng=np.random.default_rng(2321)
for repeat in range(30):
    for n in rng.permutation(state['counts']):
        n=int(n);episode=api('POST','/session',{'task':'single-spectrum'})
        token=episode['session_id'];budget=float(episode['time_budget']);assert abs(budget-30)<1e-8
        duration=budget/n;locations=np.linspace(energy.min(),energy.max(),n)
        record={'repeat':repeat+1,'count':n,'duration':duration,'session_id':token,'time_budget':budget,'observations':[]}
        state['episodes'].append(record);save()
        print(f'Start block {repeat+1}/30: N={n}, duration={duration:g}',flush=True)
        for wavelength in locations:
            result=api('POST','/pixel',{'wavelength':float(wavelength),'time':duration},token)
            record['observations'].append(result);save()
        x=np.array([r['wavelength'] for r in record['observations']]);y=np.array([r['intensity'] for r in record['observations']])
        x=(x-energy.min())/np.ptp(energy)
        model,metadata,_,_,_=fit(x,y,np.full(n,duration))
        prediction=model((energy-energy.min())/np.ptp(energy))
        assert prediction.shape==energy.shape and np.isfinite(prediction).all()
        record['fit_metadata']=metadata;record['prediction']=prediction.tolist();record['submission_attempted']=True;save()
        record['score']=api('POST','/submit',{'energy':energy.tolist(),'predicted_intensity':prediction.tolist()},token);save()
        print('Score:',json.dumps(record['score']),flush=True)
print('Peak count/duration organizer sweep complete.',flush=True)
