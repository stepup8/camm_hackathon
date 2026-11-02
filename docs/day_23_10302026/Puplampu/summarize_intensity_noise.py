import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import nnls
OUT=Path(__file__).resolve().parent
state=json.loads((OUT/'challenge1_intensity_noise_sweep.json').read_text())
assert len(state['observations'])==240
rng=np.random.default_rng(2305)
rows=[]
for energy in state['energies']:
    values=np.array([r['result']['intensity'] for r in state['observations'] if r['energy']==energy])
    assert len(values)==30 and np.isfinite(values).all()
    sd=float(values.std(ddof=1));boot=np.std(rng.choice(values,size=(10000,30),replace=True),axis=1,ddof=1)
    ci=np.quantile(boot,[.025,.975])
    rows.append({'energy':energy,'n':30,'mean_intensity':float(values.mean()),'mean_se':sd/np.sqrt(30),'noise_sd':sd,'variance':sd**2,'sd_bootstrap_95_low':float(ci[0]),'sd_bootstrap_95_high':float(ci[1])})
mean=np.array([r['mean_intensity'] for r in rows]);sd=np.array([r['noise_sd'] for r in rows]);variance=sd**2
rho,p=spearmanr(mean,sd);r,pearson_p=pearsonr(mean,variance)
coeff,_=nnls(np.column_stack([np.ones(len(mean)),np.maximum(mean,0)]),variance)
# Bootstrap Pearson variance/intensity correlation descriptively within energy groups.
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for row in rows:
    i=state['energies'].index(row['energy'])
    vals=[o['result']['intensity'] for o in state['observations'] if o['energy']==row['energy']]
    axes[0].scatter(np.full(30,i)+rng.uniform(-.15,.15,30),vals,s=15,alpha=.5)
axes[0].errorbar(range(8),mean,yerr=sd,fmt='ko',capsize=4,label='Mean ± sample SD')
axes[0].set(xticks=range(8),xticklabels=state['energies'],xlabel='Energy (a.u.)',ylabel='Measured intensity',title='30 repeats at each energy; duration = 1');axes[0].legend();axes[0].grid(alpha=.2)
axes[1].errorbar(mean,sd,xerr=[r['mean_se'] for r in rows],yerr=np.array([[r['noise_sd']-r['sd_bootstrap_95_low'] for r in rows],[r['sd_bootstrap_95_high']-r['noise_sd'] for r in rows]]),fmt='o',capsize=4)
for row in rows:axes[1].annotate(f'E={row["energy"]:g}',(row['mean_intensity'],row['noise_sd']),xytext=(4,5),textcoords='offset points',fontsize=9)
x=np.linspace(0,max(mean)*1.05,100)
axes[1].plot(x,np.sqrt(coeff[0]+coeff[1]*x),label='Nonnegative linear variance fit',ls='--')
axes[1].set(xlabel='Mean measured intensity',ylabel='Noise standard deviation',title='Does brighter signal have greater absolute noise?');axes[1].legend();axes[1].grid(alpha=.2)
fig.savefig(OUT/'challenge1_intensity_noise.png',dpi=160)
summary={'rows':rows,'spearman_mean_vs_sd':float(rho),'spearman_p_value':float(p),'pearson_mean_vs_variance':float(r),'pearson_p_value':float(pearson_p),'nonnegative_variance_fit':{'intercept':float(coeff[0]),'slope':float(coeff[1])},'pooled_within_energy_sd':float(np.sqrt(np.mean(variance))),'caveat':'Eight energies, 30 repeats each, duration 1. Correlation is across eight estimated energy-group means and noise SDs. It does not establish intensity as the cause of any energy-dependent noise. Bootstrap intervals assume independent observations; sessions may contribute variability.'}
(OUT/'challenge1_intensity_noise_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'challenge1_intensity_noise_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(summary,indent=2))
