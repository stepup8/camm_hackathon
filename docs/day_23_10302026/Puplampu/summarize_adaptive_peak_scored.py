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
state=json.loads((OUT/'challenge1_adaptive_peak_scored.json').read_text())
assert len(state['episodes'])==40 and all('score' in e for e in state['episodes'])
assert all(e['score']['accepted'] and e['score']['measurements']==30 and abs(e['score']['time_used']-30)<1e-6 for e in state['episodes'])
rows=[];values={}
for strategy in ['uniform','adaptive']:
    episodes=[e for e in state['episodes'] if e['strategy']==strategy]
    scores=np.array([e['score']['rmse'] for e in episodes]);assert len(scores)==20
    values[strategy]=scores
    rows.append({'strategy':strategy,'episodes':20,'mean_rmse':float(scores.mean()),'sample_sd':float(scores.std(ddof=1)),'median_rmse':float(np.median(scores)),'min_rmse':float(scores.min()),'max_rmse':float(scores.max()),'selected_peak_counts':dict(Counter(e['fit_metadata']['peaks'] for e in episodes)),'scores':scores.tolist()})
a=values['uniform'];b=values['adaptive'];rng=np.random.default_rng(2316)
ba=rng.choice(a,size=(100000,20),replace=True).mean(axis=1);bb=rng.choice(b,size=(100000,20),replace=True).mean(axis=1)
va=a.var(ddof=1)/20;vb=b.var(ddof=1)/20;se=np.sqrt(va+vb);df=(va+vb)**2/(va**2/19+vb**2/19);delta=float(b.mean()-a.mean())
summary={'results':rows,'mean_difference_adaptive_minus_uniform':delta,'bootstrap_95_difference':np.quantile(bb-ba,[.025,.975]).tolist(),'welch_95_difference':(delta+np.array([-1,1])*t.ppf(.975,df)*se).tolist(),'welch_two_sided_p':float(ttest_ind(b,a,equal_var=False).pvalue),'mean_rmse_reduction_percent':float(100*(1-b.mean()/a.mean())),'bootstrap_95_reduction_percent':np.quantile(100*(1-bb/ba),[.025,.975]).tolist(),'time_used':float(sum(e['score']['time_used'] for e in state['episodes'])),'caveat':'Fixed prospective test, 20 independent episodes per arm, randomized order within blocks; no paired noise realizations. Both arms use same final peak reconstruction, 30 unique locations, equal durations, and total budget 30. Bootstrap resamples whole episodes independently. Results apply to this particular adaptive rule and spectrum/API.'}
(OUT/'challenge1_adaptive_peak_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'challenge1_adaptive_peak_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (OUT/'challenge1_adaptive_peak_episode_scores.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['repeat','strategy','rmse','time_used','measurements']);writer.writeheader()
    for e in state['episodes']:writer.writerow({'repeat':e['repeat'],'strategy':e['strategy'],**{k:e['score'][k] for k in ['rmse','time_used','measurements']}})
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
for j,s in enumerate(['uniform','adaptive']):
    scores=values[s]
    axes[0].scatter(j+rng.uniform(-.12,.12,20),scores,color='steelblue',alpha=.7,s=25)
    axes[0].errorbar(j,scores.mean(),yerr=t.ppf(.975,19)*scores.std(ddof=1)/np.sqrt(20),fmt='o',color='darkorange',capsize=6)
axes[0].set(xticks=[0,1],xticklabels=['Uniform','Adaptive'],ylabel='Organizer RMSE',title='20 episodes per arm\nMean ± 95% t interval');axes[0].grid(axis='y',alpha=.2)
for s,color in [('uniform','darkorange'),('adaptive','steelblue')]:
    scores=values[s]
    axes[1].plot(range(1,21),np.cumsum(scores)/np.arange(1,21),'o-',label=s.capitalize(),color=color,markersize=3)
axes[1].set(xlabel='Episodes per arm',ylabel='Cumulative mean RMSE',title='Identical final peak reconstruction\n30 measurements, equal time');axes[1].legend();axes[1].grid(alpha=.2)
for e in state['episodes']:
    if e['strategy']!='adaptive':continue
    x=np.array([o['result']['wavelength'] for o in e['observations']]);y=np.full(30,e['repeat'])
    axes[2].scatter(x[:15],y[:15],s=8,color='steelblue')
    axes[2].scatter(x[15:],y[15:],s=8,color='darkorange')
axes[2].set(xlabel='Energy (a.u.)',ylabel='Adaptive episode',title='Adaptive locations\nBlue: initial; orange: sequential');axes[2].grid(alpha=.2)
fig.savefig(OUT/'challenge1_adaptive_peak_scored.png',dpi=160)
print(json.dumps({**summary,'results':[{k:v for k,v in r.items() if k!='scores'} for r in rows]},indent=2))
