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
state=json.loads((OUT/'challenge1_peak_variable_time_scored.json').read_text())
assert len(state['episodes'])==40 and all('score' in e for e in state['episodes'])
assert all(e['score']['accepted'] and e['score']['measurements']==30 and abs(e['score']['time_used']-30)<1e-6 for e in state['episodes'])
rows=[];values={}
for strategy in ['equal','variable']:
    episodes=[e for e in state['episodes'] if e['strategy']==strategy]
    scores=np.array([e['score']['rmse'] for e in episodes]);assert len(scores)==20;values[strategy]=scores
    for e in episodes:
        assert len(e['observations'])==30 and len(set(o['location_index'] for o in e['observations']))==30
    rows.append({'strategy':strategy,'episodes':20,'mean_rmse':float(scores.mean()),'sample_sd':float(scores.std(ddof=1)),'median_rmse':float(np.median(scores)),'min_rmse':float(scores.min()),'max_rmse':float(scores.max()),'selected_peak_counts':dict(Counter(e['fit_metadata']['peaks'] for e in episodes)),'scores':scores.tolist()})
a=values['equal'];b=values['variable'];rng=np.random.default_rng(2318)
ba=rng.choice(a,size=(100000,20),replace=True).mean(axis=1);bb=rng.choice(b,size=(100000,20),replace=True).mean(axis=1)
va=a.var(ddof=1)/20;vb=b.var(ddof=1)/20;se=np.sqrt(va+vb);df=(va+vb)**2/(va**2/19+vb**2/19);delta=float(b.mean()-a.mean())
variable_episodes=[e for e in state['episodes'] if e['strategy']=='variable']
durations=np.array([[o['requested_duration'] for o in sorted(e['observations'],key=lambda o:o['location_index'])] for e in variable_episodes])
summary={'results':rows,'mean_difference_variable_minus_equal':delta,'bootstrap_95_difference':np.quantile(bb-ba,[.025,.975]).tolist(),'welch_95_difference':(delta+np.array([-1,1])*t.ppf(.975,df)*se).tolist(),'welch_two_sided_p':float(ttest_ind(b,a,equal_var=False).pvalue),'mean_rmse_reduction_percent':float(100*(1-b.mean()/a.mean())),'bootstrap_95_reduction_percent':np.quantile(100*(1-bb/ba),[.025,.975]).tolist(),'variable_duration_range':[float(durations.min()),float(durations.max())],'variable_mean_duration_by_location':durations.mean(axis=0).tolist(),'time_used':float(sum(e['score']['time_used'] for e in state['episodes'])),'caveat':'Prospective fixed-policy comparison, 20 independent episodes per arm. Same 30 uniform locations and acquisition order, identical final peak fitting, total budget 30; no extra pilot readings. Variable policy reuses earlier independent calibration outside current episode, and updates with the current episode after 15 measurements. No common random noise seeds; bootstrap resamples episodes independently. Applies to this policy and challenge.'}
(OUT/'challenge1_peak_variable_time_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'challenge1_peak_variable_time_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (OUT/'challenge1_peak_variable_time_episode_scores.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['repeat','strategy','rmse','time_used','measurements']);writer.writeheader()
    for e in state['episodes']:writer.writerow({'repeat':e['repeat'],'strategy':e['strategy'],**{k:e['score'][k] for k in ['rmse','time_used','measurements']}})
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
for j,s in enumerate(['equal','variable']):
    scores=values[s]
    axes[0].scatter(j+rng.uniform(-.12,.12,20),scores,color='steelblue',alpha=.7,s=25)
    axes[0].errorbar(j,scores.mean(),yerr=t.ppf(.975,19)*scores.std(ddof=1)/np.sqrt(20),fmt='o',color='darkorange',capsize=6)
axes[0].set(xticks=[0,1],xticklabels=['Equal time','Adaptive time'],ylabel='Organizer RMSE',title='20 episodes per strategy\nMean ± 95% t interval');axes[0].grid(axis='y',alpha=.2)
for s,color in [('equal','darkorange'),('variable','steelblue')]:
    scores=values[s]
    axes[1].plot(range(1,21),np.cumsum(scores)/np.arange(1,21),'o-',label=s.capitalize()+' time',color=color,markersize=3)
axes[1].set(xlabel='Episodes per strategy',ylabel='Cumulative mean RMSE',title='Gaussian peak fitting\nExactly 30 readings, budget 30');axes[1].legend();axes[1].grid(alpha=.2)
for schedule in durations:axes[2].plot(state['locations'],schedule,color='steelblue',alpha=.15)
axes[2].plot(state['locations'],durations.mean(axis=0),color='steelblue',label='Mean adaptive time',lw=2)
axes[2].axhline(1,color='darkorange',ls='--',label='Equal time')
axes[2].set(xlabel='Energy (a.u.)',ylabel='Acquisition duration',title='Time allocation at fixed locations\nFaint curves: individual episodes');axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
fig.savefig(OUT/'challenge1_peak_variable_time_scored.png',dpi=160)
print(json.dumps({**summary,'results':[{k:v for k,v in row.items() if k!='scores'} for row in rows]},indent=2))
