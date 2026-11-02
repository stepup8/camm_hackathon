"""Peak reconstruction and local integrated-variance acquisition; no API calls."""
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
source=(OUT/'compare_reconstruction_methods.py').read_text()
# Reuse the exact fitted model; expose parameters for acquisition derivatives.
source=source[:source.index("report={'methods'")]
source=source.replace("'nonconverged_starts':failures}","'nonconverged_starts':failures,'parameters':best[1].tolist()}")
ns={'__file__':str(OUT/'compare_reconstruction_methods.py')}
exec(compile(source,'original_peak_helpers','exec'),ns)
A=ns['A'];B=ns['B']

def fit(x,y,time):
    order=np.argsort(x)
    x=np.asarray(x)[order];y=np.asarray(y)[order];time=np.asarray(time)[order]
    var=ns['variance'](y,time)
    model,meta=ns['peak_fit'](x,y,var)
    return model,meta,var,x,y

def jacobian(q,parameters):
    q=np.asarray(q)
    columns=[np.ones(len(q)),q-.5]
    for amplitude,center,width in np.asarray(parameters[2:]).reshape(-1,3):
        d=q-center;g=np.exp(-.5*(d/width)**2)
        columns.extend([g,amplitude*g*d/width**2,amplitude*g*d**2/width**3])
    return np.column_stack(columns)

def select_next(grid,x,y,time,adaptive_step):
    model,meta,var,x,y=fit(x,y,time)
    distances=np.min(abs(grid[:,None]-x[None,:]),axis=1)
    available=distances>=.01 # 0.1 energy unit exclusion radius on normalized axis
    if not available.any():raise RuntimeError('No available candidate locations')
    if adaptive_step%5==0:
        score=distances.copy();rule='coverage'
    else:
        train_j=jacobian(x,meta['parameters']);grid_j=jacobian(grid,meta['parameters'])
        fisher=train_j.T@(train_j/var[:,None])
        ridge=max(float(np.linalg.eigvalsh(fisher).max()),1.)*1e-8
        covariance=np.linalg.inv(fisher+ridge*np.eye(fisher.shape[0]))
        residual_scale=max(1.,float(np.sum((model(x)-y)**2/var)/max(len(y)-len(meta['parameters']),1)))
        covariance*=residual_scale
        signal_covariance=grid_j@covariance@grid_j.T
        predicted_noise=A+B*np.maximum(model(grid),0)
        denominator=np.maximum(np.diag(signal_covariance),0)+predicted_noise
        score=np.sum(signal_covariance**2,axis=0)/np.maximum(denominator,1e-12)
        rule='peak model variance reduction'
    score[~available]=-np.inf
    index=int(np.argmax(score))
    return index,rule,meta
