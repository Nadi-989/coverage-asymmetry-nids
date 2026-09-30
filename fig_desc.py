import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.linewidth':.8,
    'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':300,
    'savefig.bbox':'tight','savefig.pad_inches':.02})

C_BEN='#4C72B0'; C_ATK='#C44E52'; C_ALT='#55A868'; C_G='#8172B2'

# ---------- Fig 1 : harvest signature across datasets ----------
cic=pd.read_parquet('data/ids2018_infiltration.parquet')
doh=pd.read_parquet('data/doh_exfil_dedup.parquet')

cb=cic.loc[cic.Attack_Label==0,'TotLen Fwd Pkts'].values
ca=cic.loc[cic.Attack_Label==1,'TotLen Fwd Pkts'].values
db=doh.loc[doh.y==0,'FlowBytesSent'].values
d1=doh.loc[doh.grp=='dns2tcp','FlowBytesSent'].values
d2=doh.loc[doh.grp=='dnscat2','FlowBytesSent'].values
d3=doh.loc[doh.grp=='iodin','FlowBytesSent'].values
dapt=np.array([0,107,0,0,0,0])

data=[cb,ca,dapt,db,d1,d2,d3]
labels=['Benign','Infiltration','Exfiltration\n(n=6)','Benign-DoH','dns2tcp','DNSCat2','Iodine']
cols=[C_BEN,C_ATK,C_ATK,C_BEN,C_ATK,C_ATK,C_ATK]

fig,ax=plt.subplots(figsize=(7.0,3.0))
pos=[0,1, 2.6, 4.2,5.2,6.2,7.2]
for i,(d,p,c) in enumerate(zip(data,pos,cols)):
    v=np.log10(np.clip(d,1,None)+1)
    parts=ax.violinplot([v],positions=[p],widths=.75,showextrema=False)
    for b in parts['bodies']: b.set_facecolor(c); b.set_alpha(.55); b.set_edgecolor(c); b.set_linewidth(.8)
    ax.plot([p-.2,p+.2],[np.median(v)]*2,color='k',lw=1.4,zorder=3)
ax.set_xticks(pos); ax.set_xticklabels(labels,fontsize=8)
ax.set_ylabel(r'$\log_{10}(1+\mathrm{egress\ bytes})$')
for x,t in [(0.5,'CSE-CIC-IDS2018'),(2.6,'DAPT 2020'),(5.7,'CIRA-CIC-DoHBrw-2020')]:
    ax.text(x,7.6,t,ha='center',fontsize=8,style='italic')
for x in (1.8,3.4): ax.axvline(x,color='.75',lw=.7,ls=':')
ax.set_ylim(0,8.6)
fig.savefig('figures/fig1_signature.pdf'); fig.savefig('figures/fig1_signature.png')
print('fig1 ok')

# ---------- Fig 2 : asymmetric containment ----------
ch=doh[doh.grp=='chrome']; ff=doh[doh.grp=='firefox']; ml=doh[doh.y==1].sample(8000,random_state=0)
fig,ax=plt.subplots(figsize=(3.6,3.3))
lg=lambda s: np.log10(np.clip(s,1,None))
ax.scatter(lg(ml.FlowSentRate),lg(ml.FlowReceivedRate),s=2,c=C_ATK,alpha=.10,lw=0,label='Exfiltration')
ax.scatter(lg(ff.FlowSentRate),lg(ff.FlowReceivedRate),s=3,c=C_ALT,alpha=.30,lw=0,label='Benign (Firefox)')
ax.scatter(lg(ch.FlowSentRate),lg(ch.FlowReceivedRate),s=3,c=C_BEN,alpha=.55,lw=0,label='Benign (Chrome)')
for d,c,ls in [(ff,C_ALT,'--'),(ch,C_BEN,'-')]:
    x0,x1=lg(d.FlowSentRate).quantile([.01,.99]); y0,y1=lg(d.FlowReceivedRate).quantile([.01,.99])
    ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=c,lw=1.3,ls=ls,zorder=5))
ax.set_xlabel(r'$\log_{10}$ flow sent rate'); ax.set_ylabel(r'$\log_{10}$ flow received rate')
lg_=ax.legend(frameon=False,fontsize=7,loc='lower right',markerscale=3); 
fig.savefig('figures/fig2_containment.pdf'); fig.savefig('figures/fig2_containment.png')
print('fig2 ok')
