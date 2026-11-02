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
state = json.loads((SOURCE / 'challenge1_adaptive_checkpoint.json').read_text())
previous = json.loads((SOURCE / 'challenge1_adaptive_prediction.json').read_text())
observations = state['observations']
energy = np.asarray(state['energy'])
x = np.array([r['wavelength'] for r in observations])
y = np.array([r['intensity'] for r in observations])
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

rows=[]
models={}
for fraction in [None, .01, .03, .05, .1, .2, .3]:
    label = 'Original learned noise' if fraction is None else f'Fixed noise SD: {fraction:g} × training SD'
    errors=[]
    for i in held_out:
        indices=np.array([j for j in range(len(y)) if j!=i])
        model,_=fit(indices, fraction)
        errors.append(float(model.predict(X[i:i+1])[0]-y[i]))
    model, fit_warnings = fit(np.arange(len(y)), fraction)
    models[label]=model
    row=dict(label=label, noise_fraction=fraction,
        loo_rmse=float(np.sqrt(np.mean(np.square(errors)))),
        loo_mae=float(np.mean(np.abs(errors))),
        full_fit_noise_sd=float(np.sqrt(model.kernel_.k2.noise_level)*np.std(y)),
        length_scale_energy=float(model.kernel_.k1.k2.length_scale*np.ptp(energy)),
        kernel=str(model.kernel_), full_fit_warnings=fit_warnings,
        held_out_errors=errors)
    rows.append(row)
    print(label, 'RMSE:',round(row['loo_rmse'],6), 'MAE:',round(row['loo_mae'],6), flush=True)

linear_errors=[]
for i in held_out:
    keep=order[order!=i]
    linear_errors.append(float(np.interp(x[i], x[keep], y[keep])-y[i]))
# Pick among fixed-noise candidates only, then compare to learned-noise baseline.
best=min(rows[1:],key=lambda r:r['loo_rmse'])
best_model=models[best['label']]
prediction=best_model.predict(U)
assert np.isfinite(prediction).all() and prediction.shape==energy.shape
original=models[rows[0]['label']].predict(U)
assert np.allclose(original,previous['predicted_intensity'],rtol=1e-8,atol=1e-10)
summary=dict(results=rows, selected=best['label'],
    linear_loo_rmse=float(np.sqrt(np.mean(np.square(linear_errors)))),
    held_out_indices=held_out.tolist(),
    caveat='Hyperparameters selected on these same 28 interior leave-one-out folds. These scores are tuning diagnostics, not unbiased test errors or organizer RMSE. No hidden ground truth is available.')
(OUT/'challenge1_noise_comparison.json').write_text(json.dumps(summary,indent=2))
(OUT/'challenge1_noise_tolerant_prediction.json').write_text(json.dumps(
    dict(energy=energy.tolist(),predicted_intensity=prediction.tolist()),indent=2))

fig,axes=plt.subplots(2,1,figsize=(11,8),layout='constrained')
ax=axes[0]
ax.plot(energy,original,label='Original learned-noise GP',alpha=.8)
ax.plot(energy,prediction,label=f'Selected noise-tolerant GP (noise SD = {best["full_fit_noise_sd"]:.4f})',lw=2)
ax.scatter(x,y,s=28,c='black',label='Existing measurements',zorder=3)
ax.set(xlabel='Energy (a.u.)',ylabel='Intensity',title='Same 30 measurements — reconstruction only')
ax.legend();ax.grid(alpha=.2)
labels=['Learned']+[f'{r["noise_fraction"]:g}' for r in rows[1:]]
axes[1].bar(labels,[r['loo_rmse'] for r in rows],color=['steelblue' if r is not best else 'darkorange' for r in rows])
axes[1].set(xlabel='Assumed noise SD / training intensity SD (or learned)',ylabel='Interior leave-one-out RMSE',title='Noise-setting comparison — lower is better')
axes[1].grid(axis='y',alpha=.2)
fig.savefig(OUT/'challenge1_noise_comparison.png',dpi=160)
print('Selected:',best['label'],flush=True)
print('Full-data fit:',best['kernel'],flush=True)
print('Selected noise SD:',best['full_fit_noise_sd'],flush=True)
print('Linear LOO RMSE:',summary['linear_loo_rmse'],flush=True)
