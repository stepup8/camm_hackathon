"""Offline, within-dataset LOO comparison; never contacts organizer API."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json,warnings
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator,make_smoothing_spline
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,Matern

OUT=Path(__file__).resolve().parent
SOURCE=Path('/Users/spuplampu/camm_hackathon/day_23_10302026')
A=.0020503603314821926;B=.006151322109474143
METHODS=['Linear','PCHIP','Weighted smoothing spline','Gaussian peaks','Noise-calibrated GP']
datasets=[]
for filename in ['challenge1_variable_time_sweep.json','challenge1_variable_time_extension.json']:
    state=json.loads((OUT/filename).read_text())
    for j,e in enumerate(state['episodes']):
        if e['strategy']=='equal':
            datasets.append(dict(id=f'{filename}:{j}',group='primary_equal_two_read',x=state['locations'],y=e['combined_intensities'],time=e['allocated_total_durations'],energy=state['energy']))
def add_single(identifier,obs,energy):
    datasets.append(dict(id=identifier,group='secondary_single_read',x=[r['wavelength'] for r in obs],y=[r['intensity'] for r in obs],time=[r.get('measurement_time',1.) for r in obs],energy=energy))
state=json.loads((OUT/'challenge1_time_sweep.json').read_text())
for j,e in enumerate(state['episodes']):
    if e['count']==30:add_single(f'time_sweep:{j}',e['observations'],state['energy'])
state=json.loads((OUT/'challenge1_uniform_gp_live_checkpoint.json').read_text());add_single('uniform_gp_live',state['observations'],state['energy'])
state=json.loads((SOURCE/'challenge1_uniform_measurements.json').read_text())
datasets.append(dict(id='original_uniform',group='secondary_single_read',x=state['measured_energy'],y=state['measured_intensity'],time=[1.]*30,energy=state['energy']))
assert len(datasets)==25

def variance(y,time):
    # Fold-specific signal proxy: avoids using held-out intensities and reduces
    # weight distortion from raw noisy measurements. Calibration fixed earlier.
    pad=np.pad(y,(1,1),mode='edge')
    signal=np.array([np.median(pad[i:i+3]) for i in range(len(y))])
    return (A+B*np.maximum(signal,0))/time

def peak_fit(x,y,var):
    dense=np.linspace(0,1,256)
    rough=gaussian_filter1d(np.interp(dense,x,y),2)
    candidates,_=find_peaks(rough)
    candidates=sorted(candidates,key=lambda j:rough[j],reverse=True)
    base=float(np.clip(np.percentile(y,15),-.2,.4))
    best=None;failures=0
    for k in range(1,5):
        centers=[]
        for j in candidates:
            if all(abs(dense[j]-c)>.04 for c in centers):centers.append(float(dense[j]))
            if len(centers)==k:break
        for c in np.linspace(.1,.9,k+2):
            if len(centers)==k:break
            if all(abs(c-d)>.03 for d in centers):centers.append(float(c))
        centers=np.sort(centers[:k])
        if len(centers)!=k:raise RuntimeError('Cannot initialize peaks')
        def curve(p,q):
            value=p[0]+p[1]*(q-.5)
            for amplitude,center,width in p[2:].reshape(-1,3):value=value+amplitude*np.exp(-.5*((q-center)/width)**2)
            return value
        lower=[-.5,-1.]+[0.,0.,.015]*k
        upper=[.5,1.]+[2.,1.,.3]*k
        for initial_width in [.045,.09]:
            p=[base,0.]
            for c in centers:p.extend([float(np.clip(np.interp(c,x,y)-base,.05,1.5)),float(c),initial_width])
            fit=least_squares(lambda p:(curve(p,x)-y)/np.sqrt(var),p,bounds=(lower,upper),max_nfev=500)
            if not fit.success:
                failures+=1
                continue
            # Known-variance Gaussian likelihood; constants common across K.
            criterion=float(np.sum(fit.fun**2)+len(fit.x)*np.log(len(y)))
            if best is None or criterion<best[0]:best=(criterion,fit.x.copy(),k,curve)
    if best is None:raise RuntimeError('All peak fits failed to converge')
    return lambda q:best[3](best[1],np.asarray(q)),{'peaks':best[2],'bic_without_constant':best[0],'nonconverged_starts':failures}

def predict(method,x,y,time,q):
    var=variance(y,time)
    if method=='Linear':return np.interp(q,x,y),{}
    if method=='PCHIP':return PchipInterpolator(x,y,extrapolate=False)(q),{}
    if method=='Weighted smoothing spline':
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            spline=make_smoothing_spline(x,y,w=1/var,lam=None)
        return spline(q),{'warnings':[str(w.message) for w in caught]}
    if method=='Gaussian peaks':
        model,meta=peak_fit(x,y,var)
        return model(q),meta
    if method=='Noise-calibrated GP':
        scale=float(np.std(y)) or 1.
        gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,(1e-3,1e3))*Matern(.08,(2/511,.5),nu=1.5),
            alpha=var/scale**2,normalize_y=True,n_restarts_optimizer=3,random_state=23)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            gp.fit(x[:,None],y)
        return gp.predict(np.asarray(q).reshape(-1,1)),{'kernel':str(gp.kernel_),'warnings':[str(w.message) for w in caught]}
    raise ValueError(method)

report={'methods':METHODS,'protocol':'Interior leave-one-out (28 folds), training-fold-only weighting and model selection; standardized coordinates [0,1]. Primary cohort: 20 equal-time two-read episodes. Secondary: five single-read uniform episodes.',
'noise_calibration':{'variance_intercept':A,'variance_intensity_slope':B,'time_scaling':'inverse time approximation'},
'peak_model':'1–4 Gaussian peaks + linear background; BIC selects count within each training fold; two initial widths; bounded amplitudes, centers, widths.',
'caveat':'Predicts noisy held-out observations, not hidden noiseless ground truth. Endpoint extrapolation not tested. Estimated noise is imperfect and uses earlier API calibration. Different peak shapes could yield different results. No API calls or organizer submissions.',
'datasets':[]}
FILE=OUT/'challenge1_reconstruction_comparison.json'
for index,dataset in enumerate(datasets):
    x=np.asarray(dataset['x']);y=np.asarray(dataset['y']);time=np.asarray(dataset['time']);order=np.argsort(x)
    x=x[order]/10.;y=y[order];time=time[order]
    assert len(x)==30 and np.all(np.diff(x)>0) and np.isfinite(y).all()
    item={'id':dataset['id'],'group':dataset['group'],'methods':{},'held_out_energy':(x[1:-1]*10).tolist(),'held_out_intensity':y[1:-1].tolist()}
    for method in METHODS:
        predictions=[];metadata=[]
        for i in range(1,len(x)-1):
            keep=np.arange(len(x))!=i
            pred,meta=predict(method,x[keep],y[keep],time[keep],np.array([x[i]]))
            predictions.append(float(np.asarray(pred).ravel()[0]));metadata.append(meta)
        errors=np.array(predictions)-y[1:-1]
        full,full_meta=predict(method,x,y,time,np.asarray(dataset['energy'])/10.)
        assert np.isfinite(full).all() and np.isfinite(errors).all()
        item['methods'][method]={'rmse':float(np.sqrt(np.mean(errors**2))),'mae':float(np.mean(abs(errors))),'errors':errors.tolist(),'predictions':predictions,'full_prediction':np.asarray(full).tolist(),'full_fit':full_meta,'fold_metadata':metadata}
    report['datasets'].append(item)
    FILE.write_text(json.dumps(report,indent=2))
    print(f'{index+1}/25 {dataset["group"]}: '+', '.join(f'{m}={item["methods"][m]["rmse"]:.4f}' for m in METHODS),flush=True)
print('Offline comparison complete.',flush=True)
