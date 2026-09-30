import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.size':8.5,'font.family':'DejaVu Sans','figure.dpi':300,
 'savefig.bbox':'tight','savefig.pad_inches':.05})
OK='#4C72B0'; BAD='#C44E52'; NEU='#8C8C8C'; GRN='#55A868'

fig,ax=plt.subplots(figsize=(7.2,3.6)); ax.set_xlim(0,100); ax.set_ylim(0,52); ax.axis('off')
def box(x,y,w,h,t,s=None,c=NEU,fs=8.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.6",lw=1.1,ec=c,fc=c+'18'))
    ax.text(x+w/2,y+h/2+(1.6 if s else 0),t,ha='center',va='center',fontsize=fs,color='k')
    if s: ax.text(x+w/2,y+h/2-2.4,s,ha='center',va='center',fontsize=7,color='#444')
def arr(x1,y1,x2,y2,c='#666',st='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=.9,color=c,linestyle=st))

ax.text(0,49,'Stage 1 — audit  (§3–5)',fontsize=8.5,style='italic',color='#333')
box(0,32,20,10,'CSE-CIC-IDS2018','Infiltration',NEU)
box(0,20,20,10,'DAPT 2020','Data Exfiltration',NEU)
box(0,8,20,10,'CIRA-DoHBrw-2020','Malicious DoH',NEU)
box(26,20,16,22,'Harvest\nsignature','P1–P4  ·  R1–R4',GRN,8.5)
for y in (37,25,13): arr(20.5,y,25.5,31 if y==37 else (31 if y==25 else 31))
box(48,34,20,8,'rejected','median egress 40 B',BAD)
box(48,23,20,8,'rejected','6 flows, 5 empty',BAD)
box(48,12,20,8,'accepted','249,969 flows',OK)
arr(42.5,33,47.5,38,BAD); arr(42.5,29,47.5,27,BAD); arr(42.5,25,47.5,16,OK)

ax.text(74,49,'Stage 2 — evaluate  (§6–10)',fontsize=8.5,style='italic',color='#333')
box(74,32,25,10,'leave-attack-out','ROC 0.999',OK)
box(74,19,25,10,'+ leave-client-out','ROC 0.601',BAD)
box(74,6,25,10,'coverage asymmetry','C = 0.888 vs 0.325',GRN)
arr(68.5,16,73.5,37,OK); arr(86.5,32,86.5,29.5,'#666'); arr(86.5,19,86.5,16.5,'#666')
ax.text(86.5,30.6,'hold out the benign client too',ha='center',fontsize=6.8,color='#666')
fig.savefig('figures/fig0_overview.pdf'); fig.savefig('figures/fig0_overview.png')
print('ok')
