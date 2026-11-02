"""Prospective organizer-scored comparison, fixed uniform acquisition."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import requests
OUT=Path(__file__).resolve().parent
FILE=OUT/'challenge1_peak_vs_linear_scored.json'
BASE='https://athena-spectral-api.onrender.com'
def api(method,path,payload=None,token=None):
    response=requests.request(method,BASE+path,json=payload,headers={'X-Session-ID':token} if token else {},timeout=120)
    response.raise_for_status();return response.json()
if FILE.exists():raise RuntimeError('Existing experiment found; inspect before rerunning. No automatic retries.')
# Import only definitions and data setup, without running the offline benchmark.
source=(OUT/'compare_reconstruction_methods.py').read_text()
ns={'__file__':str(OUT/'compare_reconstruction_methods.py')}
exec(compile(source[:source.index("report={'methods'")],'offline_model_helpers','exec'),ns)
info=api('GET','/info');energy=np.linspace(*info['wavelength_range'],info['channels'])
locations=np.linspace(energy.min(),energy.max(),30)
state={'created_utc':datetime.now(timezone.utc).isoformat(),'repeats_per_method':20,'energy':energy.tolist(),'locations':locations.tolist(),
       'acquisition':'30 uniform locations, equal duration, entire episode budget',
       'peak_model':'Unchanged offline implementation: 1–4 Gaussian peaks + linear background, calibrated-variance weighted least squares, two initial widths, training-data BIC count selection',
       'noise_calibration':{'intercept':ns['A'],'slope':ns['B']},'episodes':[]}
def save():FILE.write_text(json.dumps(state,indent=2))
save();rng=np.random.default_rng(2313)
for repeat in range(20):
    for method in rng.permutation(['Linear','Gaussian peaks']):
        episode=api('POST','/session',{'task':'single-spectrum'})
        token=episode['session_id'];budget=float(episode['time_budget']);duration=budget/30
        record={'repeat':repeat+1,'method':str(method),'session_id':token,'time_budget':budget,'duration':duration,'observations':[]}
        state['episodes'].append(record);save()
        print(f'Start {repeat+1}/20: {method}',flush=True)
        for wavelength in locations:
            result=api('POST','/pixel',{'wavelength':float(wavelength),'time':duration},token)
            record['observations'].append(result);save()
        measured_x=np.array([r['wavelength'] for r in record['observations']]);y=np.array([r['intensity'] for r in record['observations']])
        order=np.argsort(measured_x);measured_x=measured_x[order];y=y[order]
        x=(measured_x-energy.min())/np.ptp(energy);q=(energy-energy.min())/np.ptp(energy)
        prediction,metadata=ns['predict'](str(method),x,y,np.full(30,duration),q)
        assert prediction.shape==energy.shape and np.isfinite(prediction).all()
        record['fit_metadata']=metadata;record['prediction']=prediction.tolist();record['submission_attempted']=True;save()
        record['score']=api('POST','/submit',{'energy':energy.tolist(),'predicted_intensity':prediction.tolist()},token);save()
        print('Score:',json.dumps(record['score']),flush=True)
print('Organizer-scored peak-versus-linear comparison complete.',flush=True)
