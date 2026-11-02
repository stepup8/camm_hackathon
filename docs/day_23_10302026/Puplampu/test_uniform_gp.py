"""Offline GP noise comparison. No API calls; run with the hackathon environment."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

SOURCE = Path('/Users/spuplampu/camm_hackathon/day_23_10302026')
OUT = Path(__file__).resolve().parent
data = json.loads((SOURCE / 'challenge1_uniform_measurements.json').read_text())
energy = np.asarray(data['energy'])
x = np.asarray(data['measured_energy'])
y = np.asarray(data['measured_intensity'])
assert len(x) == len(y) == 30 and np.isfinite(y).all()
X = ((x-energy[0]) / np.ptp(energy))[:, None]
U = ((energy-energy[0]) / np.ptp(energy))[:, None]
order = np.argsort(x)
held_out = order[1:-1]


def fit(indices, noise_fraction):
    # normalize_y uses only each training fold's values. Noise fractions are
    # standard deviations relative to that fold's observed intensity SD.
    noise = (WhiteKernel(1e-2, (1e-6, 1.0)) if noise_fraction is None
             else WhiteKernel(noise_fraction**2, 'fixed'))
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        0.08, (2/(len(energy)-1), .5), nu=1.5) + noise
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
             alpha=1e-8, n_restarts_optimizer=3, random_state=23)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(X[indices], y[indices])
    return model, [str(w.message) for w in caught]

results=[]
predictions={}
for label,fraction in [('Original GP',None),('Noise-tolerant GP',.1)]:
    errors=[]
    for i in held_out:
        indices=np.array([j for j in range(len(y)) if j!=i])
        model,_=fit(indices,fraction)
        errors.append(float(model.predict(X[i:i+1])[0]-y[i]))
    model,fit_warnings=fit(np.arange(len(y)),fraction)
    predictions[label]=model.predict(U)
    results.append(dict(model=label,loo_rmse=float(np.sqrt(np.mean(np.square(errors)))),
        loo_mae=float(np.mean(np.abs(errors))),kernel=str(model.kernel_),
        noise_sd=float(np.sqrt(model.kernel_.k2.noise_level)*np.std(y)),
        held_out_errors=errors,fit_warnings=fit_warnings))
    print(results[-1],flush=True)
errors=[]
for i in held_out:
    keep=order[order!=i]
    errors.append(float(np.interp(x[i],x[keep],y[keep])-y[i]))
predictions['Linear interpolation']=np.interp(energy,x[order],y[order])
results.append(dict(model='Linear interpolation',loo_rmse=float(np.sqrt(np.mean(np.square(errors)))),loo_mae=float(np.mean(np.abs(errors))),held_out_errors=errors))
print(results[-1],flush=True)
(OUT/'challenge1_uniform_gp_comparison.json').write_text(json.dumps(dict(results=results,held_out_indices=held_out.tolist(),caveat='Interior leave-one-out predictions of noisy measurements; not organizer RMSE. Noise-tolerant setting was previously selected using adaptive observations.'),indent=2))
for label,prediction in predictions.items():
    name={'Original GP':'original_gp','Noise-tolerant GP':'noise_tolerant_gp','Linear interpolation':'linear'}[label]
    (OUT/f'challenge1_uniform_{name}_prediction.json').write_text(json.dumps(dict(energy=energy.tolist(),predicted_intensity=prediction.tolist()),indent=2))
fig,axes=plt.subplots(2,1,figsize=(11,8),layout='constrained')
for label,prediction in predictions.items():
    axes[0].plot(energy,prediction,label=label,ls='--' if label=='Linear interpolation' else '-')
axes[0].scatter(x,y,c='black',s=25,zorder=3,label='30 uniform measurements')
axes[0].set(xlabel='Energy (a.u.)',ylabel='Intensity',title='Uniform measurements: reconstruction comparison')
axes[0].legend();axes[0].grid(alpha=.2)
adaptive=json.loads((OUT/'challenge1_noise_comparison.json').read_text())
adaptive_errors=[adaptive['results'][0]['loo_rmse'],next(r['loo_rmse'] for r in adaptive['results'] if r['noise_fraction']==.1),adaptive['linear_loo_rmse']]
positions=np.arange(3)
axes[1].bar(positions-.18,[r['loo_rmse'] for r in results],width=.36,label='Uniform locations')
axes[1].bar(positions+.18,adaptive_errors,width=.36,label='Adaptive locations')
axes[1].set(xticks=positions,xticklabels=[r['model'] for r in results],ylabel='Interior leave-one-out RMSE',title='Diagnostic only: held-out locations differ between datasets')
axes[1].legend();axes[1].grid(axis='y',alpha=.2)
fig.savefig(OUT/'challenge1_uniform_gp_comparison.png',dpi=160)
