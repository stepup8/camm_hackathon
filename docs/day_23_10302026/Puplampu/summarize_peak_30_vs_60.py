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
state=json.loads((OUT/'challenge1_peak_30_vs_60_scored.json').read_text())
assert len(state['episodes'])==60 and all('score' in e for e in state['episodes'])
assert all(e['score']['accepted'] and e['score']['measurements']==e['count'] and abs(e['score']['time_used']-30)<1e-6 for e in state['episodes'])
rows=[];values={}
for n in [30,60]:
    episodes=[e for e in state['episodes'] if e['count']==n]
    scores=np.array([e['score']['rmse'] for e in episodes]);assert len(scores)==30;values[n]=scores
    rows.append({'measurements':n,'duration':30/n,'episodes':30,'mean_rmse':float(scores.mean()),'sample_sd':float(scores.std(ddof=1)),'median_rmse':float(np.median(scores)),'min_rmse':float(scores.min()),'max_rmse':float(scores.max()),'selected_peak_counts':dict(Counter(e['fit_metadata']['peaks'] for e in episodes)),'scores':scores.tolist()})
rng=np.random.default_rng(2322)
def contrast(a,b):
    ba=rng.choice(a,size=(100000,len(a)),replace=True).mean(axis=1);bb=rng.choice(b,size=(100000,len(b)),replace=True).mean(axis=1)
    va=a.var(ddof=1)/len(a);vb=b.var(ddof=1)/len(b);se=np.sqrt(va+vb);df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1));delta=float(b.mean()-a.mean())
    return {'mean_difference_60_minus_30':delta,'bootstrap_95_difference':np.quantile(bb-ba,[.025,.975]).tolist(),'welch_95_difference':(delta+np.array([-1,1])*t.ppf(.975,df)*se).tolist(),'welch_two_sided_p':float(ttest_ind(b,a,equal_var=False).pvalue),'mean_rmse_reduction_percent':float(100*(1-b.mean()/a.mean())),'bootstrap_95_reduction_percent':np.quantile(100*(1-bb/ba),[.025,.975]).tolist()}
primary=contrast(values[30],values[60])
screen=json.loads((OUT/'challenge1_peak_count_scored.json').read_text())
pooled={n:np.concatenate([np.array([e['score']['rmse'] for e in screen['episodes'] if e['count']==n]),values[n]]) for n in [30,60]}
secondary={'label':'Exploratory pooled results: ten earlier screening episodes plus thirty fresh episodes per setting','means':{str(n):float(pooled[n].mean()) for n in [30,60]},**contrast(pooled[30],pooled[60])}
summary={'primary_results':rows,'primary_comparison':primary,'secondary_pooled':secondary,'time_used':float(sum(e['score']['time_used'] for e in state['episodes'])),'caveat':'Primary analysis uses only fresh preplanned 30 episodes per setting; screening data excluded. Same peak fitter, uniform placement, and total budget 30. Randomized setting order within blocks; noise realizations not matched. Bootstrap resamples whole episodes independently. Secondary pooled analysis includes the screening data that selected N=60 and is not independent confirmation.'}
(OUT/'challenge1_peak_30_vs_60_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'challenge1_peak_30_vs_60_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (OUT/'challenge1_peak_30_vs_60_episode_scores.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['repeat','count','duration','rmse','time_used','measurements']);writer.writeheader()
    for e in state['episodes']:writer.writerow({'repeat':e['repeat'],'count':e['count'],'duration':e['duration'],**{k:e['score'][k] for k in ['rmse','time_used','measurements']}})
fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
for j,n in enumerate([30,60]):
    scores=values[n]
    axes[0].scatter(j+rng.uniform(-.12,.12,30),scores,color='steelblue',alpha=.65,s=25)
    axes[0].errorbar(j,scores.mean(),yerr=t.ppf(.975,29)*scores.std(ddof=1)/np.sqrt(30),fmt='o',color='darkorange',capsize=6,markersize=8)
axes[0].set(xticks=[0,1],xticklabels=['30 points × 1 unit','60 points × 0.5 units'],ylabel='Organizer RMSE',title='Fresh confirmation: 30 episodes per setting\nMean ± 95% t interval');axes[0].grid(axis='y',alpha=.2)
for n,color in [(30,'darkorange'),(60,'steelblue')]:
    scores=values[n]
    axes[1].plot(range(1,31),np.cumsum(scores)/np.arange(1,31),'o-',label=f'{n} points',color=color,markersize=3)
axes[1].set(xlabel='Fresh episodes per setting',ylabel='Cumulative mean RMSE',title='Gaussian peak fitting, fixed 30-unit budget\nEarlier screening episodes excluded');axes[1].legend();axes[1].grid(alpha=.2)
fig.savefig(OUT/'challenge1_peak_30_vs_60_scored.png',dpi=160)
print(json.dumps({**summary,'primary_results':[{k:v for k,v in row.items() if k!='scores'} for row in rows]},indent=2))
