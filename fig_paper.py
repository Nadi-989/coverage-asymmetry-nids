#!/usr/bin/env python3
"""
Figures 1 and 3-9 of the revised paper, drawn from results/*.csv and data/.

Run after exp1-exp9 (run_all.sh does this). Writes figures/fig1.png, fig3.png ... fig9.png.
Figure 2 needs the CSE-CIC-IDS2018 table and is produced by fig_desc.py.
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
from common import load_doh, fit_score, default_model, subsample, TOOLS
from common import RESULTS, FIGURES
OUT=str(FIGURES)+'/'; RES=str(RESULTS)+'/'
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','savefig.pad_inches':.03})
OK='#4C72B0'; BAD='#C44E52'; NEU='#8C8C8C'; GRN='#55A868'; PUR='#8172B2'
E1=pd.read_csv(RES+'exp1_protocols.csv'); E3=pd.read_csv(RES+'exp3_containment.csv')
X19=pd.read_csv(RES+'extra_table19.csv'); P3=X19[X19.feats=='baseline'].rename(columns={'test':'browser'})
g=P3.pivot_table(index='tool',columns='browser',values='roc').reindex(TOOLS)
p2=E1[E1.protocol=='P2_leave_tool'].roc.mean(); ffx=P3[P3.browser=='firefox'].roc.mean()
cf=E3.set_index(['dataset','train']).coverage
# ---- Fig 1 overview
fig,ax=plt.subplots(figsize=(7.2,3.6)); ax.set_xlim(-2,103); ax.set_ylim(0,52); ax.axis('off')
def box(x,y,w,h,t,s=None,c=NEU,fs=8.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.6",lw=1.1,ec=c,fc=c+'18'))
    ax.text(x+w/2,y+h/2+(1.6 if s else 0),t,ha='center',va='center',fontsize=fs,weight='bold' if s else None)
    if s: ax.text(x+w/2,y+h/2-2.4,s,ha='center',va='center',fontsize=7,color='#444')
def arr(x1,y1,x2,y2,c='#666'): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=.9,color=c))
ax.text(0,49,'Stage 1 — audit (§3–5)',fontsize=8.5,style='italic',color='#333')
box(0,32,20,10,'CSE-CIC-IDS2018','Infiltration'); box(0,20,20,10,'DAPT 2020','Data Exfiltration'); box(0,8,20,10,'CIRA-CIC-DoHBrw-2020','Malicious DoH')
box(26,20,16,22,'Harvest\nsignature','P1–P4 · R1–R4',GRN)
for y in (37,25,13): arr(20.5,y,25.5,31)
box(48,34,20,8,'rejected','median egress 40 B',BAD); box(48,23,20,8,'rejected','6 flows, 5 empty',BAD); box(48,12,20,8,'accepted','249,969 flows',OK)
arr(42.5,33,47.5,38,BAD); arr(42.5,29,47.5,27,BAD); arr(42.5,25,47.5,16,OK)
ax.text(74,49,'Stage 2 — evaluate (§6–11)',fontsize=8.5,style='italic',color='#333')
box(74,32,25,10,'leave-tool-out',f'ROC-AUC {p2:.3f}',OK)
box(74,19,25,10,'+ hold out client',f'ROC-AUC {ffx:.3f} (Firefox out)',BAD)
box(74,6,25,10,'coverage asymmetry',f'C = {cf[("DoHBrw","Firefox")]:.3f} vs {cf[("DoHBrw","Chrome")]:.3f}',GRN)
arr(68.5,16,73.5,37,OK); arr(86.5,31.4,86.5,29.6); arr(86.5,18.4,86.5,16.6)
fig.savefig(OUT+'fig1.png',dpi=220)
# ---- Fig 3 grid
M=g[['chrome','firefox']].values
fig,ax=plt.subplots(figsize=(3.6,2.5)); im=ax.imshow(M,cmap='RdYlBu',vmin=.45,vmax=1,aspect='auto')
ax.set_xticks([0,1]); ax.set_xticklabels(['Chrome','Firefox']); ax.set_yticks([0,1,2]); ax.set_yticklabels(['dns2tcp','DNSCat2','Iodine'])
ax.set_xlabel('Held-out benign client'); ax.set_ylabel('Held-out tunnelling tool')
for i in range(3):
    for j in range(2): ax.text(j,i,f'{M[i,j]:.3f}',ha='center',va='center',fontsize=9,color='white' if M[i,j]<.62 else 'black')
plt.colorbar(im,ax=ax,label='ROC-AUC',fraction=.046); fig.savefig(OUT+'fig3.png',dpi=330)
# ---- Fig 5 precision/recall
grp=[('P1\nrandom',E1[E1.protocol=='P1_random']),('P2\nleave-tool',E1[E1.protocol=='P2_leave_tool']),('P3\nChrome out',P3[P3.browser=='chrome']),('P3\nFirefox out',P3[P3.browser=='firefox'])]
pr=[d.precision.mean() for _,d in grp]; rc=[d.recall.mean() for _,d in grp]
fig,ax=plt.subplots(figsize=(3.8,2.6)); x=np.arange(4); w=.36
ax.bar(x-w/2,pr,w,color=BAD,label='Precision'); ax.bar(x+w/2,rc,w,color=OK,label='Recall')
ax.set_xticks(x); ax.set_xticklabels([k for k,_ in grp],fontsize=7.5); ax.set_ylim(0,1.18); ax.set_ylabel('Score'); ax.legend(frameon=False,fontsize=7,ncol=2,loc='upper center')
for xi,(a,b) in enumerate(zip(pr,rc)): ax.text(xi-w/2,a+.02,f'{a:.3f}',ha='center',fontsize=6); ax.text(xi+w/2,b+.02,f'{b:.3f}',ha='center',fontsize=6)
fig.savefig(OUT+'fig5.png',dpi=330)
# ---- Fig 7 (asymmetry in two settings)
E2=pd.read_csv(RES+'exp2_matched_size.csv'); E5=pd.read_csv(RES+'exp5_dapt_recon.csv')
a=E2.groupby('direction').roc.agg(['mean','std']); v=[a.loc['firefox -> chrome','mean'],a.loc['chrome -> firefox','mean']]; e=[a.loc['firefox -> chrome','std'],a.loc['chrome -> firefox','std']]
b=E5.groupby('split')[['roc','f1']].mean(); v2=[b.loc['Mon+Tue->Wed','roc'],b.loc['Mon+Wed->Tue','roc']]; f2=[b.loc['Mon+Tue->Wed','f1'],b.loc['Mon+Wed->Tue','f1']]
fig,ax=plt.subplots(1,2,figsize=(7.0,3.1))
ax[0].barh([1,0],v,xerr=e,color=[OK,BAD],height=.55,error_kw=dict(lw=1,capsize=3)); ax[0].set_yticks([1,0]); ax[0].set_yticklabels(['train Firefox → test Chrome','train Chrome → test Firefox'],fontsize=7.5)
ax[0].set_title('(a) CIRA-CIC-DoHBrw-2020: held-out benign client',fontsize=8,loc='left'); ax[0].set_xlabel(f'ROC-AUC\ncontainment C = {cf[("DoHBrw","Firefox")]:.3f} vs {cf[("DoHBrw","Chrome")]:.3f}; benign training n = 3,311',fontsize=7.5)
ax[1].barh([1,0],v2,color=[OK,BAD],height=.55); ax[1].set_yticks([1,0]); ax[1].set_yticklabels(['Mon+Tue → Wed','Mon+Wed → Tue'],fontsize=7.5)
ax[1].set_title('(b) DAPT 2020: held-out capture day',fontsize=8,loc='left'); ax[1].set_xlabel(f'ROC-AUC\ncontainment C = {cf[("DAPT","Tuesday")]:.3f} vs {cf[("DAPT","Wednesday")]:.3f}',fontsize=7.5)
for A,vals,errs in ((ax[0],v,e),(ax[1],v2,[0,0])):
    A.axvline(.5,color='.4',ls=':',lw=.9); A.set_xlim(0,1.15)
    for i,val,er in zip([1,0],vals,errs): A.text(val+er+.04,i,f'{val:.3f}',va='center',fontsize=8)
for i,f in zip([1,0],f2): ax[1].text(.04,i,f'F1 = {f:.3f}',va='center',fontsize=7.5,color='white',weight='bold')
fig.tight_layout(); fig.savefig(OUT+'fig7.png',dpi=220)
# ---- Fig 8 models
D=pd.read_csv(RES+'exp7_arch_doh.csv'); P=pd.read_csv(RES+'exp7_arch_dapt.csv'); order=['XGBoost','HistGB','RandomForest','MLP','ExtraTrees','LogReg']
fig,ax=plt.subplots(1,2,figsize=(7.2,3.6)); w=.38; yv=np.arange(6)
am=D.pivot_table(index='model',columns='browser',values='roc',aggfunc='mean').reindex(order); asd=D.pivot_table(index='model',columns='browser',values='roc',aggfunc='std').reindex(order)
ax[0].barh(yv+w/2,am.chrome,w,xerr=asd.chrome,color=OK,label='covered (Chrome held out)',error_kw=dict(lw=.8,capsize=2))
ax[0].barh(yv-w/2,am.firefox,w,xerr=asd.firefox,color=BAD,label='uncovered (Firefox held out)',error_kw=dict(lw=.8,capsize=2))
ax[0].set_yticks(yv); ax[0].set_yticklabels(order,fontsize=8); ax[0].invert_yaxis(); ax[0].axvline(.5,color='.4',ls=':',lw=.9); ax[0].set_xlim(0,1.1); ax[0].set_xlabel('ROC-AUC')
ax[0].set_title('(a) DoHBrw — held-out benign client',fontsize=8.5,loc='left'); ax[0].legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.45,-.2),ncol=1)
bm=P.pivot_table(index='model',columns='split',values='f1',aggfunc='mean').reindex(order); bs=P.pivot_table(index='model',columns='split',values='f1',aggfunc='std').reindex(order)
ax[1].barh(yv+w/2,bm['Mon+Tue->Wed'],w,xerr=bs['Mon+Tue->Wed'],color=OK,error_kw=dict(lw=.8,capsize=2),label='covered (Mon+Tue → Wed)')
ax[1].barh(yv-w/2,bm['Mon+Wed->Tue'],w,xerr=bs['Mon+Wed->Tue'],color=BAD,error_kw=dict(lw=.8,capsize=2),label='uncovered (Mon+Wed → Tue)')
ax[1].set_yticks(yv); ax[1].set_yticklabels([]); ax[1].invert_yaxis(); ax[1].set_xlim(0,1.1); ax[1].set_xlabel('F1'); ax[1].set_title('(b) DAPT 2020 — held-out capture day',fontsize=8.5,loc='left'); ax[1].legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=1)
fig.tight_layout(); fig.savefig(OUT+'fig8.png',dpi=270)
# ---- Fig 9 threshold + coverage
T=pd.read_csv(RES+'exp8_threshold.csv'); xs=np.arange(len(T))
fig,ax=plt.subplots(figsize=(4.8,3.0))
ax.fill_between(xs,(T.roc-T.roc_sd).clip(upper=1),(T.roc+T.roc_sd).clip(upper=1),color=OK,alpha=.18,lw=0); ax.plot(xs,T.roc,'-o',color=OK,ms=4,lw=1.4,label='ROC-AUC (left)')
ax.axhline(T.roc[0],color=BAD,ls='--',lw=1); ax.text(.1,T.roc[0]+.012,f'no samples from target client ({T.roc[0]:.3f})',fontsize=6.8,color=BAD)
k=int(np.flatnonzero(T.n==5)[0]); ax.axvline(k,color='.5',ls=':',lw=.9); ax.text(k+.12,.66,'σ collapses at n = 5',fontsize=6.8,color='.35',rotation=90,va='bottom')
ax.set_xticks(xs); ax.set_xticklabels(T.n.astype(str)); ax.set_xlabel('benign flows from the held-out client added to training'); ax.set_ylabel('ROC-AUC'); ax.set_ylim(.55,1.02)
a2=ax.twinx(); a2.spines['right'].set_visible(True); a2.plot(xs,T.coverage,'-s',color=GRN,ms=3.5,lw=1.1,label='containment C (right)'); a2.set_ylim(0,1); a2.set_ylabel('containment C (Eq. 9)',color=GRN)
h1,l1=ax.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,frameon=False,fontsize=6.8,loc='center right',bbox_to_anchor=(1,.62))
fig.savefig(OUT+'fig9.png',dpi=280)
# ---- Fig 4 PR curves & Fig 6 scatter (need data)
df,F=load_doh(); y=df.y.values; curves={}
def fit(tr,te):
    med=df.iloc[tr][F].median(); X=df[F].fillna(med).to_numpy(np.float32); m=default_model(0).fit(X[tr],y[tr]); return m.predict_proba(X[te])[:,1],y[te]
tr,te=train_test_split(np.arange(len(df)),test_size=.3,random_state=42,stratify=y); curves['P1 random']=fit(subsample(tr,25000,0),te)
ben=np.flatnonzero(y==0); btr,bte=train_test_split(ben,test_size=.3,random_state=42,stratify=df.grp.values[ben])
mtr=subsample(np.flatnonzero(df.grp.isin(['dns2tcp','dnscat2'])),25000,0); curves['P2 leave-tool-out']=fit(np.r_[mtr,btr],np.r_[np.flatnonzero(df.grp=='iodin'),bte])
for b,o,t in [('chrome','firefox','P3 Chrome held out'),('firefox','chrome','P3 Firefox held out')]:
    tr=subsample(np.flatnonzero(df.grp.isin(['dns2tcp','dnscat2'])|(df.grp==o)),25000,0); curves[t]=fit(tr,np.r_[np.flatnonzero(df.grp=='iodin'),np.flatnonzero(df.grp==b)])
sty={'P1 random':('-',OK),'P2 leave-tool-out':('-',GRN),'P3 Chrome held out':('--',PUR),'P3 Firefox held out':('-',BAD)}
fig,ax=plt.subplots(figsize=(3.6,3.2))
for k,(p,yt) in curves.items():
    pr_,rc_,_=precision_recall_curve(yt,p); ls,c=sty[k]; ax.plot(rc_,pr_,ls,color=c,lw=1.5,label=f'{k} (AP = {average_precision_score(yt,p):.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_ylim(.55,1.02); ax.legend(frameon=False,fontsize=6.3,loc='lower left')
fig.savefig(OUT+'fig4.png',dpi=330)
ch=df[df.grp=='chrome']; ff=df[df.grp=='firefox']; ml=df[df.y==1].sample(8000,random_state=0); lg=lambda s: np.log10(np.clip(s,1,None))
fig,ax=plt.subplots(figsize=(3.6,3.3))
ax.scatter(lg(ml.FlowSentRate),lg(ml.FlowReceivedRate),s=2,c=BAD,alpha=.10,lw=0,label='Exfiltration (sample)')
ax.scatter(lg(ff.FlowSentRate),lg(ff.FlowReceivedRate),s=3,c=GRN,alpha=.30,lw=0,label='Benign (Firefox)')
ax.scatter(lg(ch.FlowSentRate),lg(ch.FlowReceivedRate),s=3,c=OK,alpha=.55,lw=0,label='Benign (Chrome)')
for d,c,ls in [(ff,GRN,'--'),(ch,OK,'-')]:
    x0,x1=lg(d.FlowSentRate).quantile([.01,.99]); y0,y1=lg(d.FlowReceivedRate).quantile([.01,.99]); ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=c,lw=1.3,ls=ls,zorder=5))
ax.set_xlabel(r'$\log_{10}$ flow sent rate'); ax.set_ylabel(r'$\log_{10}$ flow received rate'); ax.legend(frameon=False,fontsize=7,loc='lower right',markerscale=3)
fig.savefig(OUT+'fig6.png',dpi=270)
print({k:round(average_precision_score(yt,p),3) for k,(p,yt) in curves.items()})
