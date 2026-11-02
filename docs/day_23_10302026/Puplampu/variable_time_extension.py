"""Compare equal versus pilot-informed duration at 30 uniform locations."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import requests
OUT=Path(__file__).resolve().parent
FILE=OUT/'challenge1_variable_time_extension.json'
BASE='https://athena-spectral-api.onrender.com'
def api(method,path,payload=None,token=None):
    r=requests.request(method,BASE+path,json=payload,headers={'X-Session-ID':token} if token else {},timeout=120)
    r.raise_for_status();return r.json()
if FILE.exists():raise RuntimeError('Existing experiment found; inspect before rerunning. No automatic retries.')
info=api('GET','/info');energy=np.linspace(*info['wavelength_range'],info['channels'])
locations=np.linspace(energy.min(),energy.max(),30)
# Previously measured calibration; coefficients fixed before this experiment.
A=.0020503603314821926;B=.006151322109474143
PILOT=.25;MIN_TOTAL=.5;MAX_TOTAL=1.75
state={'created_utc':datetime.now(timezone.utc).isoformat(),'energy':energy.tolist(),'locations':locations.tolist(),'repeats':15,
       'variance_calibration':{'intercept':A,'slope':B},'pilot_duration':PILOT,'minimum_total_duration':MIN_TOTAL,'maximum_total_duration':MAX_TOTAL,
       'extension_of':'challenge1_variable_time_sweep.json','method':'30 fixed uniform locations; duration-weighted averages of independent readings; linear interpolation',
       'episodes':[]}
def save():FILE.write_text(json.dumps(state,indent=2))
def allocate(pilot,budget):
    # Three-point median reduces the influence of pilot noise, preserving edges.
    padded=np.pad(pilot,(1,1),mode='edge')
    signal=np.array([np.median(padded[i:i+3]) for i in range(30)])
    weights=np.sqrt(A+B*np.maximum(signal,0))
    # Fixed-location, equal-weight measurement MSE proxy gives time ∝ noise SD.
    lo,hi=0.,10000.
    for _ in range(100):
        mid=(lo+hi)/2
        if np.clip(mid*weights,MIN_TOTAL,MAX_TOTAL).sum()<budget:lo=mid
        else:hi=mid
    total=np.clip((lo+hi)/2*weights,MIN_TOTAL,MAX_TOTAL)
    total[-1]+=budget-total.sum()
    assert np.all(total>=MIN_TOTAL-1e-8) and np.all(total<=MAX_TOTAL+1e-8)
    return total,signal
save();rng=np.random.default_rng(2310)
for repeat in range(15):
    # Both arms use two reads at every location to control for splitting effects.
    for strategy in rng.permutation(['equal','variable']):
        episode=api('POST','/session',{'task':'single-spectrum'})
        token=episode['session_id'];budget=float(episode['time_budget'])
        assert abs(budget-30)<1e-8
        record={'repeat':repeat+1,'strategy':str(strategy),'session_id':token,'time_budget':budget,'observations':[]}
        state['episodes'].append(record);save()
        print(f'Start repeat {repeat+1}/15: {strategy}',flush=True)
        pilot=[]
        for i,wavelength in enumerate(locations):
            result=api('POST','/pixel',{'wavelength':float(wavelength),'time':PILOT},token)
            record['observations'].append({'location_index':i,'stage':'pilot','duration':PILOT,'result':result})
            pilot.append(result['intensity']);save()
        if strategy=='variable':total,signal=allocate(np.array(pilot),budget)
        else:total=np.full(30,budget/30);signal=np.array(pilot)
        record['allocated_total_durations']=total.tolist();record['pilot_signal_estimate']=signal.tolist();save()
        combined=[]
        for i,wavelength in enumerate(locations):
            duration=float(total[i]-PILOT)
            result=api('POST','/pixel',{'wavelength':float(wavelength),'time':duration},token)
            record['observations'].append({'location_index':i,'stage':'completion','duration':duration,'result':result})
            combined.append(float((PILOT*pilot[i]+duration*result['intensity'])/total[i]));save()
        prediction=np.interp(energy,locations,combined)
        assert np.isfinite(prediction).all()
        record['combined_intensities']=combined;record['prediction']=prediction.tolist();record['submission_attempted']=True;save()
        record['score']=api('POST','/submit',{'energy':energy.tolist(),'predicted_intensity':prediction.tolist()},token);save()
        print('Score:',json.dumps(record['score']),flush=True)
print('Variable-time sweep complete.',flush=True)
