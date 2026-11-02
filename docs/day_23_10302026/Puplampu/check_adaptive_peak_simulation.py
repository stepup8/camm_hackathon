import json
from pathlib import Path
import numpy as np
from adaptive_peak_model import fit,select_next,A,B
OUT=Path(__file__).resolve().parent
grid=np.linspace(0,1,512)
scenarios={
 'two_broad':[(.7,.3,.07),(.9,.72,.06)],
 'three_overlapping':[(.8,.24,.05),(.9,.49,.08),(.7,.72,.05)],
 'narrow_extra':[(.8,.24,.05),(.9,.5,.07),(.6,.71,.05),(.3,.88,.018)],
 'curved_background':[(.8,.23,.04),(.9,.48,.055),(.7,.74,.05)]}
results=[]
for name,peaks in scenarios.items():
    def truth(q):
        value=np.full_like(np.asarray(q),.04)+.04*np.asarray(q)
        if name=='curved_background':value=value+.03*np.sin(3*np.pi*np.asarray(q))
        for amplitude,center,width in peaks:value=value+amplitude*np.exp(-.5*((np.asarray(q)-center)/width)**2)
        return value
    for seed in range(3):
        row={'scenario':name,'seed':seed}
        for strategy in ['uniform','adaptive']:
            rng=np.random.default_rng(2400+seed);x=[];y=[]
            initial=np.linspace(0,1,30) if strategy=='uniform' else grid[np.rint(np.linspace(0,511,15)).astype(int)]
            for step in range(30):
                if step<len(initial):q=float(initial[step])
                else:
                    index,_,_=select_next(grid,np.array(x),np.array(y),np.ones(len(x)),step-14)
                    q=float(grid[index])
                value=float(truth(np.array([q]))[0]);observation=value+rng.normal()*np.sqrt(A+B*max(value,0))
                x.append(q);y.append(observation)
            model,meta,_,_,_=fit(np.array(x),np.array(y),np.ones(30))
            prediction=model(grid)
            assert len(set(x))==30 and np.isfinite(prediction).all()
            row[strategy+'_rmse']=float(np.sqrt(np.mean((prediction-truth(grid))**2)))
        results.append(row)
        print(name,seed,row['uniform_rmse'],row['adaptive_rmse'],flush=True)
(OUT/'challenge1_adaptive_peak_simulation.json').write_text(json.dumps(results,indent=2))
print('All simulation checks passed; not an organizer score.',flush=True)
