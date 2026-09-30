import pandas as pd, numpy as np, json, matplotlib, warnings; warnings.filterwarnings('ignore')
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,
 'axes.spines.right':False,'figure.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':.02})
C_BEN='#4C72B0'; C_ATK='#C44E52'; C_ALT='#55A868'

df=pd.read_parquet('data/doh_exfil_dedup.parquet')
FEAT=[c for c in df.columns if c not in ('y','grp')]; y=df.y.to_numpy()
TOOLS=['dns2tcp','dnscat2','iodin']; BRS=['chrome','firefox']
rs=np.random.RandomState(0)

def fit(tr,te):
    med=df.iloc[tr][FEAT].median(); X=df[FEAT].fillna(med).to_numpy(np.float32)
    m=HistGradientBoostingClassifier(max_iter=80,learning_rate=.15,max_bins=32,
        random_state=1,early_stopping=False,class_weight='balanced').fit(X[tr],y[tr])
    return m.predict_proba(X[te])[:,1], y[te]

curves={}
# P1 random
tr,te=train_test_split(np.arange(len(df)),test_size=.3,random_state=42,stratify=y)
tr=rs.choice(tr,60000,replace=False); curves['P1 Random']=fit(tr,te)
# P2 leave-tool-out (iodine)
ben=np.flatnonzero(df.y==0); btr,bte=train_test_split(ben,test_size=.3,random_state=42,stratify=df.grp.values[ben])
mtr=np.flatnonzero(df.grp.isin(['dns2tcp','dnscat2'])); mtr=rs.choice(mtr,50000,replace=False)
curves['P2 Leave-tool-out']=fit(np.concatenate([mtr,btr]),np.concatenate([np.flatnonzero(df.grp=='iodin'),bte]))
# P3 leave tool+browser (two directions)
for b,tag in [('chrome','P3 +client (test Chrome)'),('firefox','P3 +client (test Firefox)')]:
    ob=[x for x in BRS if x!=b][0]
    mtr=np.flatnonzero(df.grp.isin(['dns2tcp','dnscat2'])); mtr=rs.choice(mtr,50000,replace=False)
    tr=np.concatenate([mtr,np.flatnonzero(df.grp==ob)])
    te=np.concatenate([np.flatnonzero(df.grp=='iodin'),np.flatnonzero(df.grp==b)])
    curves[tag]=fit(tr,te)

# ---------- Fig 3 : PR curves ----------
fig,ax=plt.subplots(figsize=(3.6,3.2))
sty={'P1 Random':('-',C_BEN),'P2 Leave-tool-out':('-',C_ALT),
     'P3 +client (test Chrome)':('--','#8172B2'),'P3 +client (test Firefox)':('-',C_ATK)}
for k,(p,yt) in curves.items():
    pr,rc,_=precision_recall_curve(yt,p); ls,c=sty[k]
    ax.plot(rc,pr,ls,color=c,lw=1.5,label=f'{k}  (AP={average_precision_score(yt,p):.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_ylim(.55,1.02)
ax.legend(frameon=False,fontsize=6.5,loc='lower left')
fig.savefig('figures/fig3_prcurves.pdf'); fig.savefig('figures/fig3_prcurves.png')

# ---------- Fig 4 : ROC grid tool x browser ----------
M=np.array([[0.998,0.516],[0.999,0.649],[1.000,0.638]])
fig,ax=plt.subplots(figsize=(3.4,2.6))
im=ax.imshow(M,cmap='RdYlBu',vmin=.45,vmax=1.0,aspect='auto')
ax.set_xticks([0,1]); ax.set_xticklabels(['Chrome','Firefox'])
ax.set_yticks([0,1,2]); ax.set_yticklabels(['dns2tcp','DNSCat2','Iodine'])
ax.set_xlabel('Held-out benign client'); ax.set_ylabel('Held-out tunnelling tool')
for i in range(3):
    for j in range(2):
        ax.text(j,i,f'{M[i,j]:.3f}',ha='center',va='center',fontsize=9,
                color='white' if M[i,j]<.7 else 'black')
plt.colorbar(im,ax=ax,label='ROC-AUC',fraction=.046)
fig.savefig('figures/fig4_grid.pdf'); fig.savefig('figures/fig4_grid.png')

# ---------- Fig 5 : precision/recall decomposition ----------
fig,ax=plt.subplots(figsize=(3.6,2.6))
x=np.arange(3); w=.36
prec=[1.000,0.998,0.707]; rec=[1.000,0.999,0.999]
ax.bar(x-w/2,prec,w,color=C_ATK,label='Precision')
ax.bar(x+w/2,rec,w,color=C_BEN,label='Recall')
ax.set_xticks(x); ax.set_xticklabels(['P1\nRandom','P2\nLeave-tool','P3\n+client'],fontsize=8)
ax.set_ylim(0,1.15); ax.set_ylabel('Score'); ax.legend(frameon=False,fontsize=7,ncol=2,loc='upper center')
for xi,(a,b) in enumerate(zip(prec,rec)):
    ax.text(xi-w/2,a+.02,f'{a:.3f}',ha='center',fontsize=6.5)
    ax.text(xi+w/2,b+.02,f'{b:.3f}',ha='center',fontsize=6.5)
fig.savefig('figures/fig5_precrec.pdf'); fig.savefig('figures/fig5_precrec.png')
print('figs 3-5 ok')
