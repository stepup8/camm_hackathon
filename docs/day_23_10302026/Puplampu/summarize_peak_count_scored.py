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
state=json.loads((OUT/'challenge1_peak_count_scored.json').read_text())
assert len(state['episodes'])==50 and all('score' in e for e in state['episodes'])
assert all(e['score']['accepted'] and e['score']['measurements']==e['count'] and abs(e['score']['time_used']-30)<1e-6 for e in state['episodes'])
rows=[];values={};rng=np.random.default_rng(2320)
for n in state['counts']:
    episodes=[e for e in state['episodes'] if e['count']==n]
    scores=np.array([e['score']['rmse'] for e in episodes]);assert len(scores)==10;values[n]=scores
    rows.append({'measurements':n,'duration':episodes[0]['duration'],'episodes':10,'mean_rmse':float(scores.mean()),'sample_sd':float(scores.std(ddof=1)),'median_rmse':float(np.median(scores)),'min_rmse':float(scores.min()),'max_rmse':float(scores.max()),'selected_peak_counts':dict(Counter(e['fit_metadata']['peaks'] for e in episodes)),'scores':scores.tolist()})
contrasts=[]
base=values[30]
for n in state['counts']:
    if n==30:continue
    scores=values[n]
    bs=rng.choice(scores,size=(100000,10),replace=True).mean(axis=1);bb=rng.choice(base,size=(100000,10),replace=True).mean(axis=1)
    contrasts.append({'measurements':n,'mean_difference_vs_30':float(scores.mean()-base.mean()),'bootstrap_95_difference':np.quantile(bs-bb,[.025,.975]).tolist(),'mean_rmse_reduction_percent_vs_30':float(100*(1-scores.mean()/base.mean())),'welch_two_sided_p':float(ttest_ind(scores,base,equal_var=False).pvalue)})
rank=np.argsort([c['welch_two_sided_p'] for c in contrasts]);largest=0.
for position,index in enumerate(rank):
    largest=max(largest,(len(rank)-position)*contrasts[index]['welch_two_sided_p'])
    contrasts[index]['holm_adjusted_p']=float(min(largest,1.))
summary={'results':rows,'comparisons_to_30':contrasts,'lowest_mean_count':min(rows,key=lambda r:r['mean_rmse'])['measurements'],'time_used':float(sum(e['score']['time_used'] for e in state['episodes'])),'caveat':'Prospective fixed-setting count/duration sweep, ten independent episodes per setting; settings randomized within blocks. Same total budget, uniform placement, and Gaussian peak reconstruction. Four Welch comparisons versus N=30 are Holm-adjusted. Bootstrap CIs are descriptive, not multiplicity-adjusted; the lowest mean among five settings is exploratory and may be optimistic.'}
(OUT/'challenge1_peak_count_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'challenge1_peak_count_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (OUT/'challenge1_peak_count_episode_scores.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['repeat','count','duration','rmse','time_used','measurements']);writer.writeheader()
    for e in state['episodes']:writer.writerow({'repeat':e['repeat'],'count':e['count'],'duration':e['duration'],**{k:e['score'][k] for k in ['rmse','time_used','measurements']}})
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for n in state['counts']:
    scores=values[n]
    axes[0].scatter(np.full(10,n),scores,color='steelblue',alpha=.6,s=25)
axes[0].errorbar(state['counts'],[values[n].mean() for n in state['counts']],yerr=[t.ppf(.975,9)*values[n].std(ddof=1)/np.sqrt(10) for n in state['counts']],fmt='o-',color='darkorange',capsize=5,label='Mean ± 95% t interval')
axes[0].set(xticks=state['counts'],xlabel='Uniform measurements (duration = 30 / count)',ylabel='Organizer RMSE',title='Gaussian peak fitting: fixed 30-unit budget');axes[0].legend();axes[0].grid(alpha=.2)
for n in state['counts']:
    scores=values[n]
    axes[1].plot(range(1,11),np.cumsum(scores)/np.arange(1,11),'o-',label=f'N={n}, time={30/n:g}',markersize=3)
axes[1].set(xlabel='Episodes per setting',ylabel='Cumulative mean RMSE',title='Ten episodes per setting, randomized order');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
fig.savefig(OUT/'challenge1_peak_count_scored.png',dpi=160)
print(json.dumps({**summary,'results':[{k:v for k,v in row.items() if k!='scores'} for row in rows]},indent=2))
