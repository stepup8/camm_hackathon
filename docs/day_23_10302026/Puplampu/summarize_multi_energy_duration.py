import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
state=json.loads((OUT/'challenge1_multi_energy_duration_sweep.json').read_text())
assert len(state['observations'])==450
old=json.loads((OUT/'challenge1_fixed_energy_duration_sweep.json').read_text())
by_energy={e:[o for o in state['observations'] if o['energy']==e] for e in state['energies']}
by_energy[5.]=[{'duration':o['requested_duration'],'result':o['result']} for o in old['observations']]
energies=sorted(by_energy)
durations=np.array(state['durations']);rng=np.random.default_rng(2307)
B=10000; rows=[]; summaries=[]
fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
for column,e in enumerate(energies):
    samples=[np.array([o['result']['intensity'] for o in by_energy[e] if o['duration']==t]) for t in durations]
    assert all(len(v)==30 for v in samples)
    means=np.array([v.mean() for v in samples]);sd=np.array([v.std(ddof=1) for v in samples]);se=sd/np.sqrt(30)
    boots=[rng.choice(v,size=(B,30),replace=True) for v in samples]
    boot_sd=np.column_stack([v.std(axis=1,ddof=1) for v in boots]);boot_means=np.column_stack([v.mean(axis=1) for v in boots])
    logt=np.log(durations); centered=logt-logt.mean()
    slope=float(centered@np.log(sd)/sum(centered**2))
    boot_slopes=np.log(boot_sd)@centered/sum(centered**2)
    slope_ci=np.quantile(boot_slopes,[.025,.975])
    # Approximate heteroscedastic null bootstrap: preserve each duration's
    # centered residual distribution and compare weighted group means.
    weights=1/se**2;mu=float(np.sum(weights*means)/sum(weights))
    statistic=float(np.sum(weights*(means-mu)**2))
    centered_boot=boot_means-means
    boot_mu=np.sum(centered_boot*weights,axis=1)/sum(weights)
    boot_stat=np.sum(weights*(centered_boot-boot_mu[:,None])**2,axis=1)
    p=float((1+np.sum(boot_stat>=statistic))/(B+1))
    difference=float(means[-1]-means[0]); difference_ci=np.quantile(boot_means[:,-1]-boot_means[:,0],[.025,.975])
    for i,t in enumerate(durations):
        rows.append({'energy':e,'duration':float(t),'n':30,'mean':float(means[i]),'noise_sd':float(sd[i]),'mean_se':float(se[i]),'variance_times_duration':float(sd[i]**2*t)})
    summary={'energy':e,'log_sd_log_duration_slope':slope,'slope_bootstrap_95':slope_ci.tolist(),
        'variance_times_duration_mean':float(np.mean(sd**2*durations)),
        'mean_equality_bootstrap_p':p,'mean_difference_t3_minus_t025':difference,
        'mean_difference_bootstrap_95':difference_ci.tolist(),'historical_data':e==5.}
    summaries.append(summary)
    ax=axes[0,column]
    sdci=np.quantile(boot_sd,[.025,.975],axis=0)
    ax.errorbar(durations,sd,yerr=[sd-sdci[0],sdci[1]-sd],fmt='o',capsize=3,label='Observed SD')
    xx=np.geomspace(.25,3,100);c=np.exp(np.mean(np.log(sd)+.5*logt))
    ax.plot(xx,c/np.sqrt(xx),ls='--',label='1/√time reference')
    ax.set(xscale='log',yscale='log',xlabel='Duration',ylabel='Noise SD',title=f'E = {e:g}: slope {slope:.2f}'+ (' (earlier data)' if e==5 else ''))
    ax.grid(alpha=.2)
    if column==0:ax.legend(fontsize=8)
    ax=axes[1,column]
    ax.errorbar(durations,means,yerr=1.96*se,fmt='o-',capsize=3)
    ax.axhline(mu,ls='--',color='gray',alpha=.6)
    ax.set(xlabel='Duration',ylabel='Mean intensity',title='Mean ± 1.96 standard errors')
    ax.grid(alpha=.2)
# Holm correction across four descriptive tests, including earlier E=5 data.
rank=np.argsort([s['mean_equality_bootstrap_p'] for s in summaries]);maximum=0
for position,index in enumerate(rank):
    maximum=max(maximum,(len(rank)-position)*summaries[index]['mean_equality_bootstrap_p'])
    summaries[index]['mean_equality_holm_adjusted_p']=min(1.,maximum)
fig.suptitle('Duration scaling across energies: 30 observations per energy–duration pair',fontsize=15)
fig.savefig(OUT/'challenge1_multi_energy_duration.png',dpi=160)
report={'summaries':summaries,'rows':rows,'new_measurements':450,'new_time_used':sum(o['duration'] for o in state['observations']),'new_sessions':len(state['sessions']),
'caveat':'Bootstrap assumes independent observations within groups; session effects or drift may contribute. Mean-equality tests are approximate centered-residual bootstraps, Holm-adjusted across four energies. No detected mean effect is not proof of no bias. E=5 uses earlier observations; all other energies were collected together in randomized blocks. No hidden ground truth used.'}
(OUT/'challenge1_multi_energy_duration_summary.json').write_text(json.dumps(report,indent=2))
with (OUT/'challenge1_multi_energy_duration_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
