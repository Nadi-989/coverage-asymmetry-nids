import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,
 'axes.spines.right':False,'figure.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':.02})
C_OK='#4C72B0'; C_BAD='#C44E52'
D=pd.read_csv('multimodel_doh.csv'); P=pd.read_csv('multimodel_dapt.csv')
order=['XGBoost','HistGB','RandomForest','MLP','ExtraTrees','LogReg']

fig,ax=plt.subplots(1,2,figsize=(7.2,3.2))
w=.38; yv=np.arange(len(order))

a=D.pivot_table(index='model',columns='browser',values='roc',aggfunc='mean').reindex(order)
e=D.pivot_table(index='model',columns='browser',values='roc',aggfunc='std').reindex(order)
ax[0].barh(yv+w/2,a['chrome'],w,xerr=e['chrome'],color=C_OK,label='covered direction',error_kw=dict(lw=.8,capsize=2))
ax[0].barh(yv-w/2,a['firefox'],w,xerr=e['firefox'],color=C_BAD,label='uncovered direction',error_kw=dict(lw=.8,capsize=2))
ax[0].set_yticks(yv); ax[0].set_yticklabels(order,fontsize=8); ax[0].invert_yaxis()
ax[0].axvline(.5,color='.4',ls=':',lw=.9); ax[0].set_xlim(0,1.15); ax[0].set_xlabel('ROC-AUC')
ax[0].set_title('(a) DoHBrw — held-out benign client',fontsize=8.5,loc='left')
ax[0].legend(frameon=False,fontsize=7,loc='lower right')

b=P.pivot_table(index='model',columns='split',values='f1',aggfunc='mean').reindex(order)
f=P.pivot_table(index='model',columns='split',values='f1',aggfunc='std').reindex(order)
ax[1].barh(yv+w/2,b['Mon+Tue→Wed'],w,xerr=f['Mon+Tue→Wed'],color=C_OK,error_kw=dict(lw=.8,capsize=2))
ax[1].barh(yv-w/2,b['Mon+Wed→Tue'],w,xerr=f['Mon+Wed→Tue'],color=C_BAD,error_kw=dict(lw=.8,capsize=2))
ax[1].set_yticks(yv); ax[1].set_yticklabels([],fontsize=8); ax[1].invert_yaxis()
ax[1].set_xlim(0,1.15); ax[1].set_xlabel('F1')
ax[1].set_title('(b) DAPT — held-out capture day',fontsize=8.5,loc='left')
fig.savefig('figs/fig7_models.pdf'); fig.savefig('figs/fig7_models.png')
print('ok')
