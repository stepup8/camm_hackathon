import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
report=json.loads((OUT/'challenge1_reconstruction_comparison.json').read_text())
assert len(report['datasets'])==25
methods=report['methods'];rng=np.random.default_rng(2312)
rows=[];contrasts=[]
for group in ['primary_equal_two_read','secondary_single_read','all']:
    datasets=[d for d in report['datasets'] if group=='all' or d['group']==group]
    for method in methods:
        scores=np.array([d['methods'][method]['rmse'] for d in datasets])
        errors=np.concatenate([d['methods'][method]['errors'] for d in datasets])
        rows.append({'cohort':group,'method':method,'datasets':len(datasets),'mean_dataset_rmse':float(scores.mean()),'sd_dataset_rmse':float(scores.std(ddof=1)),'pooled_rmse':float(np.sqrt(np.mean(errors**2))),'pooled_mae':float(np.mean(abs(errors)))})
    for method in methods:
        if method=='Linear':continue
        delta=np.array([d['methods'][method]['rmse']-d['methods']['Linear']['rmse'] for d in datasets])
        boots=rng.choice(delta,size=(30000,len(delta)),replace=True).mean(axis=1)
        contrasts.append({'cohort':group,'method_minus_linear':method,'mean_rmse_difference':float(delta.mean()),'paired_episode_bootstrap_95':np.quantile(boots,[.025,.975]).tolist()})
primary=[d for d in report['datasets'] if d['group']=='primary_equal_two_read']
# Direct paired GP comparisons, descriptive CIs, no held-out observations pooled as independent replicates.
gp_contrasts=[]
for method in methods:
    if method=='Noise-calibrated GP':continue
    delta=np.array([d['methods'][method]['rmse']-d['methods']['Noise-calibrated GP']['rmse'] for d in primary])
    boots=rng.choice(delta,size=(30000,len(delta)),replace=True).mean(axis=1)
    gp_contrasts.append({'method':method,'mean_rmse_difference_vs_gp':float(delta.mean()),'paired_episode_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),'episodes_better_than_gp':int(np.sum(delta<0))})
summary={'results':rows,'comparisons_to_linear':contrasts,'primary_comparisons_to_gp':gp_contrasts,'caveat':report['caveat']+' Bootstrap intervals resample whole datasets; intervals are descriptive and not corrected for multiple comparisons. Methods are compared within identical measured datasets.'}
(OUT/'challenge1_reconstruction_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'challenge1_reconstruction_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
labels=['Linear','PCHIP','Weighted spline','Gaussian peaks','Noise-calibrated GP']
for j,method in enumerate(methods):
    scores=np.array([d['methods'][method]['rmse'] for d in primary])
    axes[0].scatter(j+rng.uniform(-.13,.13,len(scores)),scores,s=20,alpha=.6,color='steelblue')
    axes[0].errorbar(j,scores.mean(),yerr=scores.std(ddof=1),fmt='o',color='darkorange',capsize=5)
axes[0].set(xticks=range(5),xticklabels=labels,ylabel='Interior leave-one-out RMSE',title='Same measurements: 20 primary datasets, 28 held-out interior locations each')
axes[0].grid(axis='y',alpha=.2);axes[0].plot([],[],'o',color='steelblue',label='Dataset');axes[0].plot([],[],'o',color='darkorange',label='Mean ± sample SD');axes[0].legend()
# Prespecified example: first dataset; no selection by method performance.
example=report['datasets'][0];energy=np.linspace(0,10,512)
for method,label in zip(methods,labels):axes[1].plot(energy,example['methods'][method]['full_prediction'],label=label,alpha=.85)
state=json.loads((OUT/'challenge1_variable_time_sweep.json').read_text())
e=next(e for e in state['episodes'] if e['strategy']=='equal')
axes[1].scatter(state['locations'],e['combined_intensities'],s=22,color='black',label='Measurements',zorder=3)
axes[1].set(xlabel='Energy (a.u.)',ylabel='Intensity',title='Full fits on the first dataset — hidden ground truth unavailable')
axes[1].legend(ncol=3,fontsize=9);axes[1].grid(alpha=.2)
fig.savefig(OUT/'challenge1_reconstruction_comparison.png',dpi=160)
for group in ['primary_equal_two_read','secondary_single_read','all']:
    print(group)
    for row in sorted([r for r in rows if r['cohort']==group],key=lambda r:r['mean_dataset_rmse']):print(row)
print('PRIMARY CONTRASTS VS GP',json.dumps(gp_contrasts,indent=2))
