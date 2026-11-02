"""Budget-constrained sequential duration allocation at fixed locations."""
from pathlib import Path
import json
import numpy as np
from adaptive_peak_model import fit,A,B
OUT=Path(__file__).resolve().parent
calibration=json.loads((OUT/'challenge1_intensity_noise_summary.json').read_text())['rows']
calibration=sorted(calibration,key=lambda r:r['energy'])
CAL_E=np.array([r['energy'] for r in calibration])
CAL_MEAN=np.array([r['mean_intensity'] for r in calibration])
CAL_VAR=np.array([r['variance'] for r in calibration])
MIN_TIME=.5;MAX_TIME=1.75

def coverage_order(locations):
    chosen=[0]
    while len(chosen)<len(locations):
        distance=np.min(abs(locations[:,None]-locations[chosen][None,:]),axis=1)
        distance[chosen]=-np.inf
        chosen.append(int(np.argmax(distance)))
    assert len(set(chosen))==len(locations)
    return chosen

def bounded_allocation(weights,budget):
    n=len(weights)
    assert n>0 and n*MIN_TIME-1e-7<=budget<=n*MAX_TIME+1e-7
    if n==1:return np.array([budget])
    lo,hi=0.,10000.
    for _ in range(100):
        multiplier=(lo+hi)/2
        if np.clip(multiplier*weights,MIN_TIME,MAX_TIME).sum()<budget:lo=multiplier
        else:hi=multiplier
    allocation=np.clip((lo+hi)/2*weights,MIN_TIME,MAX_TIME)
    # Put floating-point correction at a non-boundary position.
    index=int(np.argmax(np.minimum(allocation-MIN_TIME,MAX_TIME-allocation)))
    allocation[index]+=budget-allocation.sum()
    assert np.all(allocation>=MIN_TIME-1e-7) and np.all(allocation<=MAX_TIME+1e-7)
    return allocation

def duration_plan(remaining_energy,budget,x,y,time):
    # Previously measured one-unit noise variance, interpolated in energy.
    prior_var=np.interp(remaining_energy,CAL_E,CAL_VAR)
    prior_mean=np.interp(remaining_energy,CAL_E,CAL_MEAN)
    metadata={'basis':'saved per-energy noise calibration'}
    if len(x)>=15:
        model,fit_meta,_,_,_=fit(np.asarray(x),np.asarray(y),np.asarray(time))
        predicted_mean=model(np.asarray(remaining_energy)/10)
        # Preserve calibrated energy-specific noise while updating its
        # signal-dependent component. Equal blend limits noisy-fit feedback.
        ratio=(A+B*np.maximum(predicted_mean,0))/(A+B*np.maximum(prior_mean,0))
        ratio=np.clip(ratio,.5,2.)
        prior_var=prior_var*(.5+.5*ratio)
        metadata={'basis':'saved noise profile plus current peak fit','peak_fit':fit_meta}
    durations=bounded_allocation(np.sqrt(prior_var),budget)
    return durations,metadata

if __name__=='__main__':
    locations=np.linspace(0,10,30);order=coverage_order(locations)
    for seed in range(20):
        rng=np.random.default_rng(seed);remaining=30.
        for k in range(30):
            weights=rng.uniform(.02,.2,30-k)
            durations=bounded_allocation(weights,remaining)
            remaining-=float(durations[0])
        assert abs(remaining)<1e-8
    print('Budget and duration bounds verified over 20 sequential schedules.')
