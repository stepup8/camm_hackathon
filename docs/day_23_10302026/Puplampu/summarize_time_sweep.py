import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/day23-matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
state=json.loads((OUT/'challenge1_time_sweep.json').read_text())
rows=[]
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
for count in state['counts']:
    episodes=[e for e in state['episodes'] if e['count']==count and 'score' in e]
    assert len(episodes)==state['repeats'] and all(e['score']['accepted'] for e in episodes)
    scores=np.array([e['score']['rmse'] for e in episodes])
    assert all(e['score']['measurements']==count and abs(e['score']['time_used']-30)<1e-6 for e in episodes)
    row={'measurements':count,'duration':episodes[0]['duration'],'mean_rmse':float(scores.mean()),'sample_sd':float(scores.std(ddof=1)),'scores':scores.tolist()}
    rows.append(row)
    ax.scatter([count]*len(scores),scores,color='steelblue',alpha=.65,zorder=3)
ax.errorbar([r['measurements'] for r in rows],[r['mean_rmse'] for r in rows],yerr=[r['sample_sd'] for r in rows],fmt='o-',capsize=5,color='darkorange',label='Mean ± sample SD (3 episodes)')
ax.set(xlabel='Uniform measurements per episode (duration = 30 / count)',ylabel='Organizer RMSE — lower is better',title='Challenge 1: sampling density versus duration, fixed budget of 30',xscale='log')
ax.set_xticks(state['counts'],[str(n) for n in state['counts']]);ax.grid(alpha=.2);ax.legend()
fig.savefig(OUT/'challenge1_time_sweep.png',dpi=160)
(OUT/'challenge1_time_sweep_summary.json').write_text(json.dumps(rows,indent=2))
with (OUT/'challenge1_time_sweep_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for r in rows:print(r)
