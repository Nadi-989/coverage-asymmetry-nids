#!/usr/bin/env python3
"""
Experiment 9 - supplementary results (Sections 6.2, 9.3-9.4; Appendix B, Table B2).

  * Table 19 in both directions, 20 seeds x 3 held-out tools (n = 60 per cell)
  * invariant features added to the baseline (Section 9.4)
  * P3 without the Duration feature (Section 6.2)
  * training / inference cost of the reference detector (Table B2)

Writes results/extra_table19.csv.
"""
import numpy as np, pandas as pd, time, platform, sklearn, xgboost, scipy
from scipy.stats import wilcoxon
from common import *
from exp6_invariant_features import add_invariant
df,BASE=load_doh(); CI=add_invariant(df)
rows=[]
for br in BROWSERS:
    other=[b for b in BROWSERS if b!=br][0]
    for tool in TOOLS:
        tr_all=np.flatnonzero(df.grp.isin([t for t in TOOLS if t!=tool])|(df.grp==other))
        te=np.flatnonzero((df.grp==tool)|(df.grp==br))
        for s in range(20):
            tr=subsample(tr_all,25_000,s)
            for name,F in [("baseline",BASE),("invariant",CI)]+([("combined",BASE+CI)] if br=="firefox" else [])+([("no_duration",[f for f in BASE if f!="Duration"])] if s<2 else []):
                rows.append(dict(test=br,tool=tool,seed=s,feats=name,**fit_score(df,F,tr,te,seed=s)))
R=pd.DataFrame(rows); R.to_csv(RESULTS/"extra_table19.csv",index=False)
print(R.groupby(['feats','test'])[['roc','pr']].agg(['mean','std']).round(4))
ff=R[R.test=="firefox"]
b=ff[ff.feats=="baseline"].sort_values(["tool","seed"]); c=ff[ff.feats=="combined"].sort_values(["tool","seed"])
d=c.roc.values-b.roc.values; print("combined-baseline ROC",d.mean().round(4),d.std(ddof=1).round(4),wilcoxon(c.roc.values,b.roc.values).pvalue)
nd=R[R.feats=="no_duration"]; bd=R[(R.feats=="baseline")&(R.seed<2)]
print("P3 all, baseline(2 seeds) vs no-duration:",bd.roc.mean().round(4),nd.roc.mean().round(4)); print(nd.groupby('test').roc.mean().round(4), bd.groupby('test').roc.mean().round(4))
# timing
tr_all=np.flatnonzero(df.grp.isin(["dns2tcp","dnscat2"])|(df.grp=="firefox")); tr=subsample(tr_all,25_000,0)
med=df.iloc[tr][BASE].median(); X=df[BASE].fillna(med).to_numpy(np.float32); y=df.y.values
ts=[]
for s in range(3):
    m=default_model(s); t=time.perf_counter(); m.fit(X[tr],y[tr]); ts.append(time.perf_counter()-t)
te=np.flatnonzero(df.grp=="iodin")[:20000]
t=time.perf_counter(); m.predict_proba(X[te]); bt=(time.perf_counter()-t)/len(te)
one=[];
for i in te[:500]:
    t=time.perf_counter(); m.predict_proba(X[i:i+1]); one.append(time.perf_counter()-t)
import pickle; sz=len(pickle.dumps(m))
print(f"train {np.mean(ts):.2f}s +- {np.std(ts):.2f} on {len(tr)} flows; batch {bt*1e6:.2f} us/flow; single median {np.median(one)*1e3:.3f} ms p99 {np.percentile(one,99)*1e3:.3f} ms; model {sz/1024:.0f} KB")
print(platform.python_version(), sklearn.__version__, xgboost.__version__, pd.__version__, np.__version__, scipy.__version__)
