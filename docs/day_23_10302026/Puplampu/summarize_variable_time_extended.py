import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind,t
OUT=Path(__file__).resolve().parent
original=json.loads((OUT/'challenge1_variable_time_sweep.json').read_text())
extension=json.loads((OUT/'challenge1_variable_time_extension.json').read_text())
assert len(extension['episodes'])==30 and all('score' in e for e in extension['episodes'])
for key in ['locations','pilot_duration','minimum_total_duration','maximum_total_duration','variance_calibration']:
    assert original[key]==extension[key],key
all_episodes=original['episodes']+extension['episodes']
assert all(e['score']['accepted'] and e['score']['measurements']==60 and abs(e['score']['time_used']-30)<1e-6 for e in all_episodes)
rows=[];values={}
for strategy in ['equal','variable']:
    scores=np.array([e['score']['rmse'] for e in all_episodes if e['strategy']==strategy]);values[strategy]=scores
    assert len(scores)==20
    rows.append({'strategy':strategy,'episodes':len(scores),'mean_rmse':float(scores.mean()),'sample_sd':float(scores.std(ddof=1)),'median_rmse':float(np.median(scores)),'min_rmse':float(scores.min()),'max_rmse':float(scores.max()),'scores':scores.tolist()})
a=values['equal'];b=values['variable'];delta=float(b.mean()-a.mean())
rng=np.random.default_rng(2311);B=100000
ba=rng.choice(a,size=(B,20),replace=True).mean(axis=1);bb=rng.choice(b,size=(B,20),replace=True).mean(axis=1)
boot_ci=np.quantile(bb-ba,[.025,.975]);percent_ci=np.quantile(100*(1-bb/ba),[.025,.975])
va=a.var(ddof=1)/len(a);vb=b.var(ddof=1)/len(b);se=np.sqrt(va+vb)
df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1));ci=delta+np.array([-1,1])*t.ppf(.975,df)*se
test=ttest_ind(b,a,equal_var=False)
batches={}
for label,episodes in [('original_5',original['episodes']),('additional_15',extension['episodes'])]:
    batches[label]={s:float(np.mean([e['score']['rmse'] for e in episodes if e['strategy']==s])) for s in ['equal','variable']}
summary={'results':rows,'mean_difference_variable_minus_equal':delta,'bootstrap_95_difference':boot_ci.tolist(),'welch_95_difference':ci.tolist(),'welch_two_sided_p':float(test.pvalue),'mean_rmse_reduction_percent':float(100*(1-b.mean()/a.mean())),'bootstrap_95_reduction_percent':percent_ci.tolist(),'batch_means':batches,'additional_time_used':float(sum(e['score']['time_used'] for e in extension['episodes'])),'caveat':'20 independent episodes per strategy; no common random seeds or paired-noise control. Bootstrap intervals resample episodes independently. Both arms use 60 readings at 30 locations and 30 time units. Findings apply to this API, reconstruction, and allocation policy; they do not establish a globally optimal strategy.'}
(OUT/'challenge1_variable_time_extended_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'challenge1_variable_time_extended_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (OUT/'challenge1_variable_time_all_episode_scores.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['batch','repeat','strategy','rmse','time_used','measurements']);writer.writeheader()
    for batch,episodes in [('original',original['episodes']),('additional',extension['episodes'])]:
        for e in episodes:writer.writerow({'batch':batch,'repeat':e['repeat'],'strategy':e['strategy'],**{k:e['score'][k] for k in ['rmse','time_used','measurements']}})
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for j,s in enumerate(['equal','variable']):
    scores=values[s]
    axes[0].scatter(j+rng.uniform(-.12,.12,20),scores,color='steelblue',alpha=.7,s=30)
    half=t.ppf(.975,19)*scores.std(ddof=1)/np.sqrt(20)
    axes[0].errorbar(j,scores.mean(),yerr=half,fmt='o',color='darkorange',capsize=6,markersize=8)
axes[0].set(xticks=[0,1],xticklabels=['Equal time','Variable time'],ylabel='Organizer RMSE',title='20 episodes per strategy; mean ± 95% t interval');axes[0].grid(axis='y',alpha=.2)
axes[0].plot([],[],'o',color='steelblue',label='Episode');axes[0].plot([],[],'o',color='darkorange',label='Mean');axes[0].legend()
for s,color in [('equal','darkorange'),('variable','steelblue')]:
    running=np.cumsum(values[s])/np.arange(1,21)
    axes[1].plot(range(1,21),running,'o-',label=s.capitalize()+' time',color=color,markersize=3)
axes[1].axvline(5.5,color='gray',ls='--',alpha=.6,label='Additional repeats begin')
axes[1].set(xlabel='Episodes per strategy',ylabel='Cumulative mean organizer RMSE',title='Original five plus fifteen additional episodes');axes[1].legend();axes[1].grid(alpha=.2)
fig.savefig(OUT/'challenge1_variable_time_extended.png',dpi=160)
print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))
for r in rows:print({k:v for k,v in r.items() if k!='scores'})
