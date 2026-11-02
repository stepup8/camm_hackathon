import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
OUT=Path(__file__).resolve().parent
state=json.loads((OUT/'challenge1_fixed_energy_duration_sweep.json').read_text())
assert len(state['observations'])==150
rng=np.random.default_rng(2302)
rows=[]
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for j,t in enumerate(state['durations']):
    records=[o for o in state['observations'] if o['requested_duration']==t]
    values=np.array([o['result']['intensity'] for o in records])
    assert len(values)==30 and np.isfinite(values).all()
    variance=float(values.var(ddof=1));sd=variance**.5
    # Bootstrap intervals avoid assuming Gaussian measurement noise.
    bootstrap_sd=np.std(rng.choice(values,size=(10000,len(values)),replace=True),axis=1,ddof=1)
    interval=np.quantile(bootstrap_sd,[.025,.975])
    rows.append({'duration':t,'n':len(values),'mean_intensity':float(values.mean()),'sample_sd':sd,'sd_bootstrap_95_low':float(interval[0]),'sd_bootstrap_95_high':float(interval[1]),'variance_times_duration':variance*t})
    axes[0].scatter(np.full(len(values),j)+rng.uniform(-.15,.15,len(values)),values,s=22,alpha=.6)
axes[0].errorbar(np.arange(5),[r['mean_intensity'] for r in rows],yerr=[r['sample_sd'] for r in rows],fmt='ko',capsize=5,label='Mean ± sample SD')
axes[0].set(xticks=np.arange(5),xticklabels=state['durations'],xlabel='Acquisition duration',ylabel='Measured intensity',title='Repeated observations at energy 5.0');axes[0].legend();axes[0].grid(alpha=.2)
t=np.array([r['duration'] for r in rows]);sd=np.array([r['sample_sd'] for r in rows]);slope,intercept=np.polyfit(np.log(t),np.log(sd),1)
axes[1].errorbar(t,sd,yerr=np.array([[r['sample_sd']-r['sd_bootstrap_95_low'] for r in rows],[r['sd_bootstrap_95_high']-r['sample_sd'] for r in rows]]),fmt='o',capsize=5,label='SD; bootstrap 95% intervals')
x=np.geomspace(t.min(),t.max(),100)
axes[1].plot(x,np.exp(intercept)*x**slope,label=f'Fitted scaling: SD ∝ time^{slope:.2f}')
c=np.exp(np.mean(np.log(sd)+.5*np.log(t)))
axes[1].plot(x,c/np.sqrt(x),ls='--',label='Reference: SD ∝ 1/√time')
axes[1].set(xscale='log',yscale='log',xlabel='Acquisition duration',ylabel='Sample standard deviation',title='Noise versus acquisition duration');axes[1].legend();axes[1].grid(alpha=.2)
fig.savefig(OUT/'challenge1_fixed_energy_noise.png',dpi=160)
# Session offsets: fit common mean plus session offsets, inspect only descriptively.
sessions=[]
for i,s in enumerate(state['sessions']):
    records=[o for o in state['observations'] if o['session_index']==i]
    vals=np.array([o['result']['intensity'] for o in records])
    sessions.append({'index':i,'n':len(vals),'mean':float(vals.mean()),'durations':sorted(set(o['requested_duration'] for o in records))})
summary={'energy':state['energy'],'rows':rows,'log_sd_log_duration_slope':float(slope),'session_summaries':sessions,'total_acquisition_time':sum(o['requested_duration'] for o in state['observations']),'session_count':len(state['sessions']),'caveat':'One energy, 30 observations per duration across randomized blocks and multiple sessions. Bootstrap intervals are descriptive and assume independent observations; session effects or drift may affect variability.'}
(OUT/'challenge1_fixed_energy_noise_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'challenge1_fixed_energy_noise_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(summary,indent=2))
