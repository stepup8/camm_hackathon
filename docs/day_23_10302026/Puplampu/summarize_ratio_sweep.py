import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
state=json.loads((OUT/'challenge1_ratio_sweep.json').read_text())
rows=[]
fig,axes=plt.subplots(2,1,figsize=(10,8),layout='constrained')
for j,n in enumerate(state['initial_counts']):
    episodes=[e for e in state['episodes'] if e['initial_count']==n and 'score' in e]
    assert len(episodes)==3
    assert all(e['score']['accepted'] and e['score']['measurements']==30 and abs(e['score']['time_used']-30)<1e-6 for e in episodes)
    scores=np.array([e['score']['rmse'] for e in episodes])
    gaps=[float(np.max(np.diff(np.sort([o['wavelength'] for o in e['observations']])))) for e in episodes]
    row={'initial':n,'adaptive':30-n,'mean_rmse':float(scores.mean()),'sample_sd':float(scores.std(ddof=1)),'min_rmse':float(scores.min()),'max_rmse':float(scores.max()),'scores':scores.tolist(),'mean_max_gap':float(np.mean(gaps))}
    rows.append(row)
    axes[0].scatter(np.full(3,j),scores,color='steelblue',zorder=3)
    axes[0].errorbar(j,scores.mean(),yerr=scores.std(ddof=1),fmt='o',color='darkorange',capsize=6)
    for k,e in enumerate(episodes):
        x=[o['wavelength'] for o in e['observations']]
        y=np.full(30,j*4+k)
        axes[1].scatter(x[:n],y[:n],color='steelblue',s=15)
        axes[1].scatter(x[n:],y[n:],color='darkorange',s=15)
axes[0].set(xticks=range(4),xticklabels=[f'{n}–{30-n}' for n in state['initial_counts']],xlabel='Initial evenly spaced – sequential measurements',ylabel='Organizer RMSE',title='Initial/adaptive split: three episodes per setting, same 30-unit budget')
axes[0].plot([],[],'o',color='steelblue',label='Individual episode')
axes[0].plot([],[],'o',color='darkorange',label='Mean ± sample SD');axes[0].legend();axes[0].grid(alpha=.2)
axes[1].set(xlabel='Energy (a.u.)',ylabel='Split and repeat',yticks=[j*4+1 for j in range(4)],yticklabels=[f'{n}–{30-n}' for n in state['initial_counts']],title='Measurement locations: blue initial, orange sequential')
axes[1].grid(axis='x',alpha=.2)
fig.savefig(OUT/'challenge1_ratio_sweep.png',dpi=160)
(OUT/'challenge1_ratio_sweep_summary.json').write_text(json.dumps(rows,indent=2))
with (OUT/'challenge1_ratio_sweep_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for r in rows:print(r)
