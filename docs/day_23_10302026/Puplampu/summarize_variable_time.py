import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
state=json.loads((OUT/'challenge1_variable_time_sweep.json').read_text())
assert len(state['episodes'])==10 and all('score' in e for e in state['episodes'])
rows=[]
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for j,strategy in enumerate(['equal','variable']):
    episodes=[e for e in state['episodes'] if e['strategy']==strategy]
    assert len(episodes)==5
    assert all(e['score']['accepted'] and abs(e['score']['time_used']-30)<1e-6 and e['score']['measurements']==60 for e in episodes)
    scores=np.array([e['score']['rmse'] for e in episodes])
    rows.append({'strategy':strategy,'episodes':len(episodes),'mean_rmse':float(scores.mean()),'sample_sd':float(scores.std(ddof=1)),'scores':scores.tolist()})
    axes[0].scatter(np.full(5,j),scores,color='steelblue')
    axes[0].errorbar(j,scores.mean(),yerr=scores.std(ddof=1),fmt='o',color='darkorange',capsize=6)
axes[0].set(xticks=[0,1],xticklabels=['Equal time','Variable time'],ylabel='Organizer RMSE',title='Same 30 locations and 30-unit budget')
axes[0].plot([],[],'o',color='steelblue',label='Episode');axes[0].plot([],[],'o',color='darkorange',label='Mean ± sample SD');axes[0].legend();axes[0].grid(alpha=.2)
episodes=[e for e in state['episodes'] if e['strategy']=='variable']
durations=np.array([e['allocated_total_durations'] for e in episodes]);locations=np.array(state['locations'])
for values in durations:axes[1].plot(locations,values,color='steelblue',alpha=.25)
axes[1].plot(locations,durations.mean(axis=0),color='steelblue',lw=2,label='Variable time: mean across episodes')
axes[1].axhline(1,color='darkorange',ls='--',label='Equal time')
axes[1].set(xlabel='Energy (a.u.)',ylabel='Total duration per location',title='Pilot-informed allocation (pilot time included)');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
fig.savefig(OUT/'challenge1_variable_time_comparison.png',dpi=160)
change=(rows[1]['mean_rmse']/rows[0]['mean_rmse']-1)*100
# Independent-arm bootstrap, descriptive; no assumption that episode seeds pair.
rng=np.random.default_rng(2309)
a=np.array(rows[0]['scores']);b=np.array(rows[1]['scores'])
diff=rng.choice(b,size=(20000,5),replace=True).mean(axis=1)-rng.choice(a,size=(20000,5),replace=True).mean(axis=1)
summary={'results':rows,'variable_vs_equal_mean_rmse_percent_change':float(change),'difference_variable_minus_equal_bootstrap_95':np.quantile(diff,[.025,.975]).tolist(),
'actual_variable_duration_range':[float(durations.min()),float(durations.max())],
'caveat':'Five independent episodes per arm; descriptive bootstrap may be unstable with this sample size. Both arms use 60 scalar readings at 30 unique locations (pilot plus completion). All pilot time counts in the 30-unit budget. Duration weighting assumes approximately inverse-time variance; pilot-dependent allocation can introduce selection effects. This evaluates the full practical strategy, not ideal allocation with known noiseless intensities.'}
(OUT/'challenge1_variable_time_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'challenge1_variable_time_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(summary,indent=2))
