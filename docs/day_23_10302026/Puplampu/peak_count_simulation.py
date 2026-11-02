"""Fixed-budget count sweep on known synthetic spectra; not API scores."""
import json
from pathlib import Path
import numpy as np
from adaptive_peak_model import fit,A,B
OUT=Path(__file__).resolve().parent
grid=np.linspace(0,1,512)
scenarios={
 'two_broad':[(.7,.3,.07),(.9,.72,.06)],
 'three_overlapping':[(.8,.24,.05),(.9,.49,.08),(.7,.72,.05)],
 'narrow_extra':[(.8,.24,.05),(.9,.5,.07),(.6,.71,.05),(.3,.88,.018)],
 'curved_background':[(.8,.23,.04),(.9,.48,.055),(.7,.74,.05)]}
rows=[]
for name,peaks in scenarios.items():
    def truth(q):
        q=np.asarray(q);value=np.full_like(q,.04)+.04*q
        if name=='curved_background':value+=.03*np.sin(3*np.pi*q)
        for amplitude,center,width in peaks:value+=amplitude*np.exp(-.5*((q-center)/width)**2)
        return value
    for n in [15,20,30,40,60]:
        for seed in range(5):
            rng=np.random.default_rng(2500+seed);x=np.linspace(0,1,n);duration=30/n
            clean=truth(x);y=clean+rng.normal(size=n)*np.sqrt((A+B*np.maximum(clean,0))/duration)
            model,meta,_,_,_=fit(x,y,np.full(n,duration));prediction=model(grid)
            assert np.isfinite(prediction).all()
            rows.append({'scenario':name,'count':n,'seed':seed,'duration':duration,'rmse':float(np.sqrt(np.mean((prediction-truth(grid))**2))),'selected_peaks':meta['peaks']})
    print(name,'complete',flush=True)
(OUT/'challenge1_peak_count_simulation.json').write_text(json.dumps(rows,indent=2))
for n in [15,20,30,40,60]:print('N',n,'synthetic mean RMSE',np.mean([r['rmse'] for r in rows if r['count']==n]),flush=True)
print('All synthetic count/duration checks passed.',flush=True)
