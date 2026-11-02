"""Exactly 30 measurements: uniform locations, equal versus adaptive time."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import requests
from adaptive_peak_model import fit
from peak_variable_time_model import coverage_order,duration_plan,CAL_E,CAL_MEAN,CAL_VAR,MIN_TIME,MAX_TIME
OUT=Path(__file__).resolve().parent
FILE=OUT/'challenge1_peak_variable_time_scored.json'
BASE='https://athena-spectral-api.onrender.com'
def api(method,path,payload=None,token=None):
    response=requests.request(method,BASE+path,json=payload,headers={'X-Session-ID':token} if token else {},timeout=120)
    response.raise_for_status();return response.json()
if FILE.exists():raise RuntimeError('Existing experiment found; inspect before rerunning. No automatic retries.')
info=api('GET','/info');energy=np.linspace(*info['wavelength_range'],info['channels'])
assert np.allclose(energy[[0,-1]],[0,10])
locations=np.linspace(0,10,30);order=coverage_order(locations)
state={'created_utc':datetime.now(timezone.utc).isoformat(),'repeats_per_strategy':20,'energy':energy.tolist(),'locations':locations.tolist(),'location_order':order,
'acquisition':'Exactly 30 readings at fixed uniform locations in shared coverage order. Equal time control versus bounded noise-SD-proportional time; all readings count in episode budget.',
'calibration':{'energy':CAL_E.tolist(),'mean_intensity':CAL_MEAN.tolist(),'one_unit_variance':CAL_VAR.tolist(),'source':'earlier independent per-energy noise experiment'},
'adaptive_rule':'Start from saved variance profile; after 15 readings, update signal-dependent variance by 50% blend with current peak-model correction; correction ratio bounded [0.5,2]. Reallocate remaining budget after every reading. Total durations bounded [0.5,1.75].',
'reconstruction':'Same original BIC-selected Gaussian peak fitter for both arms','episodes':[]}
def save():FILE.write_text(json.dumps(state,indent=2))
save();rng=np.random.default_rng(2317)
for repeat in range(20):
    for strategy in rng.permutation(['equal','variable']):
        episode=api('POST','/session',{'task':'single-spectrum'})
        token=episode['session_id'];budget=float(episode['time_budget']);assert abs(budget-30)<1e-8
        record={'repeat':repeat+1,'strategy':str(strategy),'session_id':token,'time_budget':budget,'observations':[]}
        state['episodes'].append(record);save()
        print(f'Start {repeat+1}/20: {strategy}',flush=True)
        x=[];y=[];times=[];remaining=budget
        for step,index in enumerate(order):
            if strategy=='equal':duration=remaining/(30-step);meta={}
            else:
                plan,meta=duration_plan(locations[order[step:]],remaining,x,y,times)
                duration=float(plan[0])
            assert MIN_TIME-1e-7<=duration<=MAX_TIME+1e-7
            wavelength=float(locations[index])
            result=api('POST','/pixel',{'wavelength':wavelength,'time':duration},token)
            x.append(float(result['wavelength'])/10);y.append(float(result['intensity']));times.append(duration)
            remaining=budget-sum(times)
            record['observations'].append({'location_index':index,'requested_duration':duration,'allocation_metadata':meta,'result':result});save()
        assert abs(sum(times)-budget)<1e-7 and len(set(x))==30
        model,metadata,_,_,_=fit(np.array(x),np.array(y),np.array(times))
        prediction=model(energy/10)
        assert prediction.shape==energy.shape and np.isfinite(prediction).all()
        record['fit_metadata']=metadata;record['prediction']=prediction.tolist();record['submission_attempted']=True;save()
        record['score']=api('POST','/submit',{'energy':energy.tolist(),'predicted_intensity':prediction.tolist()},token);save()
        print('Score:',json.dumps(record['score']),flush=True)
print('Peak fitting with adaptive measurement time: comparison complete.',flush=True)
