"""One fresh uniform-placement episode, reconstructed by the original GP."""
import json
from pathlib import Path
import numpy as np
import requests
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

OUT=Path(__file__).resolve().parent
CHECKPOINT=OUT/'challenge1_uniform_gp_live_checkpoint.json'
BASE='https://athena-spectral-api.onrender.com'
def api(method,path,payload=None,token=None):
    r=requests.request(method,BASE+path,json=payload,
        headers={'X-Session-ID':token} if token else {},timeout=120)
    r.raise_for_status()
    return r.json()
def save(state):
    CHECKPOINT.write_text(json.dumps(state,indent=2))
if CHECKPOINT.exists():
    raise RuntimeError('An episode checkpoint already exists. Inspect it before any rerun; no automatic retries.')
info=api('GET','/info')
energy=np.linspace(*info['wavelength_range'],info['channels'])
assert np.all(np.diff(energy)>0)
episode=api('POST','/session',{'task':'single-spectrum'})
state={'session_id':episode['session_id'],'energy':energy.tolist(),
       'time_budget':float(episode['time_budget']),'observations':[]}
save(state)
token=state['session_id']
duration=state['time_budget']/30
for i,wavelength in enumerate(np.linspace(energy.min(),energy.max(),30)):
    result=api('POST','/pixel',{'wavelength':float(wavelength),'time':duration},token)
    state['observations'].append(result)
    save(state)
    print(f'Measurement {i+1}/30 saved',flush=True)
x=np.array([r['wavelength'] for r in state['observations']])
y=np.array([r['intensity'] for r in state['observations']])
kernel=ConstantKernel(1.,(1e-3,1e3))*Matern(.08,(2/(len(energy)-1),.5),nu=1.5)+WhiteKernel(.01,(1e-6,1.))
gp=GaussianProcessRegressor(kernel=kernel,normalize_y=True,alpha=1e-8,n_restarts_optimizer=3,random_state=23)
gp.fit(((x-energy.min())/np.ptp(energy))[:,None],y)
prediction=gp.predict(((energy-energy.min())/np.ptp(energy))[:,None])
assert prediction.shape==energy.shape and np.isfinite(prediction).all()
submission={'energy':energy.tolist(),'predicted_intensity':prediction.tolist()}
(OUT/'challenge1_uniform_gp_live_prediction.json').write_text(json.dumps(submission,indent=2))
state['kernel']=str(gp.kernel_)
state['budget_before_submission']=api('GET','/budget',token=token)
state['submission_attempted']=True
save(state)
score=api('POST','/submit',submission,token)
state['score']=score
save(state)
(OUT/'challenge1_uniform_gp_live_score.json').write_text(json.dumps(score,indent=2))
print('Organizer score:',json.dumps(score),flush=True)
print('Fitted kernel:',gp.kernel_,flush=True)
