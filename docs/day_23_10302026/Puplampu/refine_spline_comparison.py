"""Training-only smoothing selection on a broad log grid using linear-smoother LOO."""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import make_smoothing_spline
OUT=Path(__file__).resolve().parent
# Load function definitions and dataset construction without executing benchmark loop.
source=(OUT/'compare_reconstruction_methods.py').read_text()
ns={'__file__':str(OUT/'compare_reconstruction_methods.py')}
exec(compile(source[:source.index("report={'methods'")],'comparison_helpers','exec'),ns)
report=json.loads((OUT/'challenge1_reconstruction_comparison.json').read_text())
raw=OUT/'challenge1_reconstruction_default_spline_comparison.json'
if not raw.exists():raw.write_text(json.dumps(report,indent=2))
def spline_predict(x,y,time,q):
    variance=ns['variance'](y,time)
    weights=1/variance;weights=weights/weights.mean()
    best=None
    for lam in np.geomspace(1e-10,1e1,60):
        smoother=make_smoothing_spline(x,np.eye(len(x)),w=weights,lam=float(lam))(x)
        fit=smoother@y
        denominator=1-np.diag(smoother)
        if np.any(denominator<1e-8):continue
        score=float(np.mean(((y-fit)/denominator)**2))
        if best is None or score<best[0]:best=(score,float(lam),float(np.trace(smoother)))
    assert best is not None
    prediction=make_smoothing_spline(x,y,w=weights,lam=best[1])(q)
    return prediction,{'selected_lambda':best[1],'inner_linear_smoother_loo_mse':best[0],'effective_degrees_of_freedom':best[2],'selection':'60 log-spaced lambda values [1e-10, 10], training-only approximate inner LOO with fixed training-derived weights'}
for j,(dataset,item) in enumerate(zip(ns['datasets'],report['datasets'])):
    assert dataset['id']==item['id']
    order=np.argsort(dataset['x']);x=np.asarray(dataset['x'])[order]/10;y=np.asarray(dataset['y'])[order];time=np.asarray(dataset['time'])[order]
    predictions=[];metadata=[]
    for i in range(1,29):
        keep=np.arange(30)!=i
        pred,meta=spline_predict(x[keep],y[keep],time[keep],[x[i]])
        predictions.append(float(pred[0]));metadata.append(meta)
    errors=np.array(predictions)-y[1:-1]
    full,meta=spline_predict(x,y,time,np.asarray(dataset['energy'])/10)
    assert np.isfinite(errors).all() and np.isfinite(full).all()
    previous=item['methods']['Weighted smoothing spline']
    item['methods']['Weighted smoothing spline']={'rmse':float(np.sqrt(np.mean(errors**2))),'mae':float(np.mean(abs(errors))),'errors':errors.tolist(),'predictions':predictions,'full_prediction':full.tolist(),'full_fit':meta,'fold_metadata':metadata,'default_gcv_rmse':previous.get('default_gcv_rmse',previous['rmse'])}
    print(f'{j+1}/25 spline RMSE: {item["methods"]["Weighted smoothing spline"]["rmse"]:.5f}',flush=True)
report['spline_selection']='Default automatic GCV produced strong oversmoothing and is archived separately. Final spline comparison uses training-only inner linear-smoother LOO on a broad fixed logarithmic lambda grid. Fold-specific weights are held fixed during this inner tuning; outer held-out intensities are never used.'
(OUT/'challenge1_reconstruction_comparison.json').write_text(json.dumps(report,indent=2))
