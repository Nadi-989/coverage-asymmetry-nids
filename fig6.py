import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,
 'axes.spines.right':False,'figure.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':.02})
C_OK='#4C72B0'; C_BAD='#C44E52'

fig,ax=plt.subplots(1,2,figsize=(7.0,3.1))

# (a) DoHBrw — benign client asymmetry
lab=['Firefox → Chrome','Chrome → Firefox']
val=[0.9959,0.6612]; err=[0.0037,0.0085]
b=ax[0].barh([1,0],val,xerr=err,color=[C_OK,C_BAD],height=.55,error_kw=dict(lw=1,capsize=3))
ax[0].set_yticks([1,0]); ax[0].set_yticklabels(lab,fontsize=8)
ax[0].axvline(.5,color='.4',ls=':',lw=.9)
ax[0].set_xlim(0,1.08); ax[0].set_xlabel('ROC-AUC')
ax[0].set_title('(a) CIRA-CIC-DoHBrw-2020\nheld-out benign client',fontsize=8.5,loc='left')
for i,v in zip([1,0],val): ax[0].text(v+.03,i,f'{v:.3f}',va='center',fontsize=8)
ax[0].text(0.0,-0.75,'containment 88.8% vs 32.5%',fontsize=7.5,color='.35')

# (b) DAPT — attack-mix asymmetry
lab2=['Tue → Wed','Wed → Tue']
val2=[0.9923,0.8711]; f1=[0.9272,0.3689]
ax[1].barh([1,0],val2,color=[C_OK,C_BAD],height=.55)
ax[1].set_yticks([1,0]); ax[1].set_yticklabels(lab2,fontsize=8)
ax[1].axvline(.5,color='.4',ls=':',lw=.9)
ax[1].set_xlim(0,1.08); ax[1].set_xlabel('ROC-AUC')
ax[1].set_title('(b) DAPT 2020\nheld-out capture day',fontsize=8.5,loc='left')
for i,(v,f) in zip([1,0],zip(val2,f1)):
    ax[1].text(v+.03,i,f'{v:.3f}',va='center',fontsize=8)
    ax[1].text(.03,i-.30,f'F1 = {f:.3f}',fontsize=7.5,color='.35')
fig.savefig('figs/fig6_asymmetry.pdf'); fig.savefig('figs/fig6_asymmetry.png')
print('ok')
