"""Prospective adaptive versus uniform placement, identical final peak models."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import requests
from adaptive_peak_model import fit,select_next
OUT=Path(__file__).resolve().parent
FILE=OUT/'challenge1_adaptive_peak_scored.json'
BASE='https://athena-spectral-api.onrender.com'
def api(method,path,payload=None,token=None):
    response=requests.request(method,BASE+path,json=payload,headers={'X-Session-ID':token} if token else {},timeout=120)
    response.raise_for_status();return response.json()
if FILE.exists():raise RuntimeError('Existing experiment found; inspect before rerunning. No automatic retries.')
info=api('GET','/info');energy=np.linspace(*info['wavelength_range'],info['channels']);grid=(energy-energy.min())/np.ptp(energy)
state={'created_utc':datetime.now(timezone.utc).isoformat(),'repeats_per_strategy':20,'energy':energy.tolist(),
'acquisition':'30 equal-duration measurements; adaptive starts with 15 uniform points; 15 sequential acquisitions; every fifth sequential point fills a gap; minimum candidate distance 0.1 energy units',
'criterion':'Integrated local parameter-linearized peak-model variance reduction, calibrated observation noise, regularized Fisher covariance',
'reconstruction':'Same original BIC-selected 1–4 Gaussian peaks plus linear background for both strategies',
'episodes':[]}
def save():FILE.write_text(json.dumps(state,indent=2))
save();rng=np.random.default_rng(2315)
for repeat in range(20):
    for strategy in rng.permutation(['uniform','adaptive']):
        episode=api('POST','/session',{'task':'single-spectrum'})
        token=episode['session_id'];budget=float(episode['time_budget']);duration=budget/30
        record={'repeat':repeat+1,'strategy':str(strategy),'session_id':token,'duration':duration,'time_budget':budget,'observations':[]}
        state['episodes'].append(record);save()
        print(f'Start {repeat+1}/20: {strategy}',flush=True)
        initial=grid[np.rint(np.linspace(0,len(grid)-1,15)).astype(int)]
        x=[];y=[]
        for step in range(30):
            if strategy=='uniform':q=float(np.linspace(0,1,30)[step]);rule='uniform';meta={}
            elif step<15:q=float(initial[step]);rule='initial coverage';meta={}
            else:
                index,rule,meta=select_next(grid,np.array(x),np.array(y),np.full(len(x),duration),step-14)
                q=float(grid[index])
            wavelength=float(energy.min()+q*np.ptp(energy))
            result=api('POST','/pixel',{'wavelength':wavelength,'time':duration},token)
            observed_q=(float(result['wavelength'])-energy.min())/np.ptp(energy)
            x.append(float(observed_q));y.append(float(result['intensity']))
            record['observations'].append({'rule':rule,'requested_wavelength':wavelength,'acquisition_model':meta,'result':result});save()
        model,metadata,_,_,_=fit(np.array(x),np.array(y),np.full(30,duration))
        prediction=model(grid)
        assert prediction.shape==energy.shape and np.isfinite(prediction).all() and len(set(x))==30
        record['fit_metadata']=metadata;record['prediction']=prediction.tolist();record['submission_attempted']=True;save()
        record['score']=api('POST','/submit',{'energy':energy.tolist(),'predicted_intensity':prediction.tolist()},token);save()
        print('Score:',json.dumps(record['score']),flush=True)
print('Adaptive-peak organizer comparison complete.',flush=True)
