import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json,csv
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind,t
OUT=Path(__file__).resolve().parent
state=json.loads((OUT/'challenge1_peak_vs_linear_scored.json').read_text())
assert len(state['episodes'])==40 and all('score' in e for e in state['episodes'])
assert all(e['score']['accepted'] and e['score']['measurements']==30 and abs(e['score']['time_used']-30)<1e-6 for e in state['episodes'])
rows=[];values={}
for method in ['Linear','Gaussian peaks']:
    scores=np.array([e['score']['rmse'] for e in state['episodes'] if e['method']==method]);assert len(scores)==20
    values[method]=scores
    rows.append({'method':method,'episodes':20,'mean_rmse':float(scores.mean()),'sample_sd':float(scores.std(ddof=1)),'median_rmse':float(np.median(scores)),'min_rmse':float(scores.min()),'max_rmse':float(scores.max()),'scores':scores.tolist()})
a=values['Linear'];b=values['Gaussian peaks'];rng=np.random.default_rng(2314)
ba=rng.choice(a,size=(100000,20),replace=True).mean(axis=1);bb=rng.choice(b,size=(100000,20),replace=True).mean(axis=1)
va=a.var(ddof=1)/20;vb=b.var(ddof=1)/20;se=np.sqrt(va+vb);df=(va+vb)**2/(va**2/19+vb**2/19);delta=b.mean()-a.mean()
summary={'results':rows,'mean_difference_peaks_minus_linear':float(delta),'bootstrap_95_difference':np.quantile(bb-ba,[.025,.975]).tolist(),'welch_95_difference':(delta+np.array([-1,1])*t.ppf(.975,df)*se).tolist(),'welch_two_sided_p':float(ttest_ind(b,a,equal_var=False).pvalue),'mean_rmse_reduction_percent':float(100*(1-b.mean()/a.mean())),'bootstrap_95_reduction_percent':np.quantile(100*(1-bb/ba),[.025,.975]).tolist(),'selected_peak_counts':dict(Counter(e['fit_metadata']['peaks'] for e in state['episodes'] if e['method']=='Gaussian peaks')),'time_used':float(sum(e['score']['time_used'] for e in state['episodes'])),'caveat':'Prospective comparison with fixed settings, 20 episodes per method, randomized order in blocks. Separate noise realizations, no common random seeds; bootstrap resamples episodes independently. Organizer RMSE uses hidden clean spectrum. Finding applies to this spectrum/API and chosen model, not arbitrary spectral shapes.'}
(OUT/'challenge1_peak_vs_linear_scored_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'challenge1_peak_vs_linear_scored_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (OUT/'challenge1_peak_vs_linear_episode_scores.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['repeat','method','rmse','time_used','measurements']);writer.writeheader()
    for e in state['episodes']:writer.writerow({'repeat':e['repeat'],'method':e['method'],**{k:e['score'][k] for k in ['rmse','time_used','measurements']}})
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for j,method in enumerate(['Linear','Gaussian peaks']):
    scores=values[method]
    axes[0].scatter(j+rng.uniform(-.12,.12,20),scores,color='steelblue',alpha=.7)
    axes[0].errorbar(j,scores.mean(),yerr=t.ppf(.975,19)*scores.std(ddof=1)/np.sqrt(20),fmt='o',color='darkorange',capsize=6)
axes[0].set(xticks=[0,1],xticklabels=['Linear interpolation','Gaussian peak fitting'],ylabel='Organizer RMSE',title='20 episodes per method; mean ± 95% t interval');axes[0].grid(axis='y',alpha=.2)
for method,color in [('Linear','darkorange'),('Gaussian peaks','steelblue')]:
    scores=values[method]
    axes[1].plot(range(1,21),np.cumsum(scores)/np.arange(1,21),'o-',label=method,color=color,markersize=3)
axes[1].set(xlabel='Episodes per method',ylabel='Cumulative mean organizer RMSE',title='Same 30 uniform locations and 30-unit budget');axes[1].legend();axes[1].grid(alpha=.2)
fig.savefig(OUT/'challenge1_peak_vs_linear_scored.png',dpi=160)
print(json.dumps(summary,indent=2))
