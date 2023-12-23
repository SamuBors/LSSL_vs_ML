#%% libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import statistics
from datetime import datetime
import statsmodels.api as sm
import os
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression
from tabulate import tabulate
import matplotlib.patches as mpatches
import seaborn as sns

#%% directory
os.chdir('/Users/samueleborsini/Library/Mobile Documents/com~apple~CloudDocs/Università/Economics and econometrics/II anno/Machine Learning/Project python')

#%% importing cleaned dataset
final=pd.read_csv('Data/final.csv')
final['sasdate']= [datetime.strptime(date_str, '%Y-%m-%d') for date_str in final['sasdate']]
final.index=final['sasdate']
final=final.drop('sasdate',axis=1)

#%% importing AR(1) forecasts
y_ar_og=pd.read_csv('Data/y_ar.csv')
y_ar_og.columns=['sasdate',0]
y_ar_og['sasdate']= [datetime.strptime(date_str, '%Y-%m-%d') for date_str in y_ar_og['sasdate']]
y_ar_og.index=y_ar_og['sasdate']
y_ar_og=y_ar_og.drop('sasdate',axis=1)
y_ar_og=y_ar_og.squeeze()

#%% number of rolling windows
T=final.shape[0] #total number of observations
skip=1 #skip in predictions
rolling_window_length=120 #months in a rolling window (i.e. rows of the dataset used as training)
n_rolling_windows=T-skip-rolling_window_length #number of rolling window

#%% several rolling estimates
K_pcr=[1,3,5,10,25,50,75]
K_pls=[1,3,5,10,25,50,75]
L_ridge=10**np.linspace(-1,2,7)
L_lasso=10**np.linspace(-1,2,7)
L_lasso_loop=L_lasso/(rolling_window_length*final.shape[1])
y_pred={}
y_pred['PCR'] = pd.DataFrame(columns=K_pcr, index=range(0, n_rolling_windows))
y_pred['PLS'] = pd.DataFrame(columns=K_pls, index=range(0, n_rolling_windows))
y_pred['Ridge'] = pd.DataFrame(columns=L_ridge, index=range(0, n_rolling_windows))
y_pred['Lasso'] = pd.DataFrame(columns=L_lasso, index=range(0, n_rolling_windows))
y_dates=[]
y_true=[]
y_rw=[]
y_std=[]
zeros={}
cols=list(final.columns)
cols.append('const')
for l in L_lasso:
    zeros[l]=pd.DataFrame(columns=cols, index=range(0, n_rolling_windows))

for i in range(0,n_rolling_windows):
    #preparing the training set
    x_train=final.iloc[i:rolling_window_length+i]
    x_means=x_train.mean()
    x_stds=x_train.std()
    x_train=pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test=(final.iloc[rolling_window_length+i]-x_means)/x_stds
    y_dates.append(final.iloc[rolling_window_length+i+skip].name)
    y_train=final[dep].iloc[i+skip:rolling_window_length+i+skip]
    
    #PCR
    S=x_train.T.dot(x_train)/x_train.shape[0]
    evals, evecs = np.linalg.eig(S)
    for k in K_pcr:
        PCs=x_train@evecs.T[:k].T
        X=sm.add_constant(PCs)
        X.index=y_train.index
        model = sm.OLS(y_train, X).fit()
        PCs_test=x_test@evecs.T[:k].T
        y_pred_i=model.params['const']+PCs_test @ model.params[1:k+1]
        y_pred['PCR'].at[i,k]=y_pred_i.real
        
    #PLS
    for k in K_pls:
        pls_model = PLSRegression(n_components=k)
        pls_model.fit(x_train.values, y_train.values)
        y_pred['PLS'].at[i,k] = pls_model.predict(x_test.values.reshape(1, -1))[0][0]
    
    #Ridge
    for l in L_ridge:
        X=sm.add_constant(x_train)
        ridge_coefficients=np.linalg.inv(X.T.dot(X)+l*np.eye(X.shape[1])) @ np.dot(X.T.values,y_train.values)
        y_pred['Ridge'].at[i,l]=ridge_coefficients[0]+x_test @ ridge_coefficients[1:]
 
    #Lasso
    for l in L_lasso:
        lasso_model = Lasso(alpha=l/(rolling_window_length*final.shape[1]), tol=0.005)
        lasso_model.fit(x_train, y_train)
        y_pred['Lasso'].at[i,l] = lasso_model.intercept_ + lasso_model.coef_ @ x_test
        zeros[l].loc[i,:][final.columns]=lasso_model.coef_
        zeros[l]['const'].iloc[i]=lasso_model.intercept_
    
    #true series, random walk, sds
    y_true.append(final[dep].iloc[rolling_window_length+i+skip])
    y_rw.append(final[dep].iloc[rolling_window_length+i])
    y_std.append(y_train.std())
    print(i)
y_true=pd.Series(y_true)
y_rw=pd.Series(y_rw)
for name in list(y_pred.keys()):
    y_pred[name].index=y_dates
for l in L_lasso:
    zeros[l].index=y_dates
y_true.index=y_dates
y_rw.index=y_dates
y_ar=y_ar_og[y_ar_og.index>=y_true.index.min()]
y_std=pd.Series(y_std)
y_std.index=y_dates

#%% MSFE
SFE_rw=(y_rw-y_true)**2
MSFE_rw=SFE_rw.mean()
SFE_ar=(y_ar-y_true)**2
MSFE_ar=SFE_ar.mean()
SFE={}
MSFE={}
MSFE_relative_rw={}
MSFE_relative_ar={}
for name in y_pred.keys():
    SFE[name]=(y_pred[name].sub(y_true, axis=0))**2
    MSFE[name]=SFE[name].mean()
    MSFE_relative_rw[name]=MSFE[name]/MSFE_rw
    MSFE_relative_ar[name]=MSFE[name]/MSFE_ar

#%% prediction correlation
corr={}
for name in ['Lasso','Ridge','PLS']:
    A=y_pred[name].copy()
    A['PCR']=y_pred['PCR'][MSFE['PCR'][MSFE['PCR']==MSFE['PCR'].min()].index].squeeze()
    corr[name]=A.corr()['PCR'].drop('PCR').values

#%% subperiods MSFE
sub={}
sub[0]=[datetime(1971,1,1),datetime(1984,12,31)]
sub[1]=[datetime(1985,1,1),datetime(2003,12,31)]
sub[2]=[datetime(2004,1,1),datetime(2019,12,31)]
s=0

MSFE_subperiods={}
MSFE_subperiods['AR']={}
MSFE_subperiods['RW']={}
for s in list(sub.keys()):
    MSFE_rw_sub=SFE_rw[(SFE_rw.index>=sub[s][0]) & (SFE_rw.index<sub[s][1])].mean()
    MSFE_ar_sub=SFE_ar[(SFE_rw.index>=sub[s][0]) & (SFE_ar.index<sub[s][1])].mean()
    SFE_sub={}
    MSFE_sub={}
    MSFE_relative_rw_sub={}
    MSFE_relative_ar_sub={}
    for name in list(y_pred.keys()):
        SFE_sub[name]=(y_pred[name][(SFE_rw.index>=sub[s][0]) & (SFE_rw.index<sub[s][1])].sub(y_true[(SFE_rw.index>=sub[s][0]) & (SFE_rw.index<sub[s][1])], axis=0))**2
        MSFE_sub[name]=SFE_sub[name].mean()
        MSFE_relative_rw_sub[name]=MSFE_sub[name]/MSFE_rw_sub
        MSFE_relative_ar_sub[name]=MSFE_sub[name]/MSFE_ar_sub
    MSFE_subperiods['AR'][s]=MSFE_relative_ar_sub
    MSFE_subperiods['RW'][s]=MSFE_relative_rw_sub
    
#%% latex tables
tables_data={}
PC_opt=y_pred['PCR'][MSFE['PCR'][MSFE['PCR']==MSFE['PCR'].min()].index].columns[0]
for name in ['Lasso','Ridge','PLS']:
    tables_data[name]=[
        ['']+MSFE_subperiods['AR'][0][name].index.values.tolist(),
        ['1971-1984']+MSFE_subperiods['AR'][0][name].values.tolist(),
        ['1985-2003']+MSFE_subperiods['AR'][1][name].values.tolist(),
        ['2004-2019']+MSFE_subperiods['AR'][2][name].values.tolist(),
        ['1971-2019']+MSFE_relative_ar[name].values.tolist(),
        [f'Correlation with PCR ({PC_opt} PCs)']+corr[name].tolist()
        ]
    
name='PCR'
tables_data[name]=[
    ['']+MSFE_subperiods['AR'][0][name].index.values.tolist(),
    ['1971-1984']+MSFE_subperiods['AR'][0][name].values.tolist(),
    ['1985-2003']+MSFE_subperiods['AR'][1][name].values.tolist(),
    ['2004-2019']+MSFE_subperiods['AR'][2][name].values.tolist(),
    ['1971-2019']+MSFE_relative_ar[name].values.tolist(),
    ]

latex_tables={}
for name in list(y_pred.keys()):
    latex_tables[name] = tabulate(tables_data[name], tablefmt="latex")
    with open(f'Tables/{name}.tex', "w") as f:
        f.write(latex_tables[name])

#%% Relative MSFE plot
for name in list(y_pred.keys()):
    plt.figure(figsize=(10, 6))
    if name == 'Lasso' or name == 'Ridge':
        plt.plot(np.log10(MSFE_relative_ar[name].index),MSFE_relative_ar[name], linestyle='-', color = "red", label='MSFE relative to the AR(1)')
        plt.plot(np.log10(MSFE_relative_rw[name].index),MSFE_relative_rw[name], linestyle='-', color = "deepskyblue", label='MSFE relative to the random walk')
    else:
        plt.plot(MSFE_relative_ar[name].index,MSFE_relative_ar[name], linestyle='-', color = "red", label='MSFE relative to the AR(1)')
        plt.plot(MSFE_relative_rw[name].index,MSFE_relative_rw[name], linestyle='-', color = "deepskyblue", label='MSFE relative to the random walk')
    plt.title(f'Relative MSFE for {name}')
    if name == 'Lasso' or name == 'Ridge':
        plt.xlabel(r'$\log _{10}(\lambda)$')
    elif name == 'PLS':
        plt.xlabel('Components')
    else:
        plt.xlabel('Principal components')
    plt.ylabel('MSFE')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'Plots/MSFEs/MSFE_relative_{name}.png', dpi=500)
    plt.show()

#%% plot
for name in list(y_pred.keys()):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.index,y_true, linestyle='-', color = "black", label='True series')
    plt.plot(y_pred[name].index,y_pred[name][MSFE[name][MSFE[name]==MSFE[name].min()].index], linestyle='--', color = "mediumvioletred", label=name, linewidth=0.9)
    plt.plot(y_ar.index,y_ar, linestyle='--', color = 'deepskyblue', label='AR(1)', linewidth=0.9)
    plt.title(f'True series, AR(1) and {name} forecasts')
    if name == 'Lasso' or name == 'Ridge':
        plt.title(rf'True series, AR(1) and {name} forecasts (with $\lambda$={MSFE[name][MSFE[name]==MSFE[name].min()].index[0]})')
    elif name == 'PLS':
        plt.title(f'True series, AR(1) and {name} forecasts (with {MSFE[name][MSFE[name]==MSFE[name].min()].index[0]} components)')
    else:
        plt.title(f'True series, AR(1) and {name} forecasts (with {MSFE[name][MSFE[name]==MSFE[name].min()].index[0]} principal components)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'Plots/Forecasts/{name}_forecasts.png', dpi=500)
    plt.show()

#%% MSFE AR vs. MSFE
for name in list(y_pred.keys()):
    plt.figure(figsize=(10, 6))
    plt.plot(SFE_ar.index,SFE_ar, linestyle='-', color = "deepskyblue", label='AR(1)')
    plt.plot(SFE[name].index,SFE[name][MSFE[name][MSFE[name]==MSFE[name].min()].index], linestyle='-', color = "mediumvioletred", label=name)
    if name == 'Lasso' or name == 'Ridge':
        plt.title(f'MSFE AR(1) vs. MSFE {name} (with $\lambda$={MSFE[name][MSFE[name]==MSFE[name].min()].index[0]})')
    elif name == 'PLS':
        plt.title(f'MSFE AR(1) vs. MSFE {name} (with {MSFE[name][MSFE[name]==MSFE[name].min()].index[0]} components)')
    else:
        plt.title(f'MSFE AR(1) vs. MSFE {name} (with {MSFE[name][MSFE[name]==MSFE[name].min()].index[0]} principal components)')
    plt.xlabel('Time')
    plt.ylabel('MSFE')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'Plots/MSFEs/MSFE_AR_vs_MSFE_{name}.png', dpi=500)
    plt.show()

#%% LASSO coefficients
zeros_opt=zeros[MSFE['Lasso'][MSFE['Lasso']==MSFE['Lasso'].min()].index[0]]
zeros_opt_1_0=(zeros_opt!=0).astype(int)
SP=zeros_opt_1_0.mean()
SP_t15=SP.nlargest(10).sort_values(ascending=False)
SP_t15.plot.bar(rot=45)
plt.title('Percentage of times the coefficient has not been shrunk')
plt.tight_layout()
plt.savefig('Plots/shirnkage_percentage.png', dpi=500)

#%% LASSO graph
zeros_opt_10=zeros_opt_1_0.T
plt.figure(figsize=(10, 6))
cmap = sns.color_palette(['grey', 'red'])
sns.heatmap(zeros_opt_10, annot=False, cmap=cmap, fmt=".2f", cbar=False, yticklabels=False, xticklabels=False)
plt.xlabel('Time')
xtick_positions = np.array([107, 227, 347, 467])
xtick_labels = ['1980', '1990', '2000', '2010']
plt.xticks(xtick_positions, xtick_labels)
plt.ylabel('Predictors')
plt.title('Lasso coefficients shrinkage', loc='left')
red_patch = mpatches.Patch(color='red', label='Coefficient not shrunk')
grey_patch = mpatches.Patch(color='grey', label='Coefficient shrunk')
plt.legend(handles=[red_patch, grey_patch], bbox_to_anchor=(1.008,1.0725), loc='upper right', ncol=2)
plt.savefig('Plots/lasso_coefs.png', dpi=500)
plt.show()

#%% overperforming percentage
OP_ar=[]
OP_rw=[]
index=[]
for name in list(y_pred.keys()):
    OP_ar.append((SFE[name][MSFE[name][MSFE[name]==MSFE[name].min()].index].values.ravel()<SFE_ar.values).mean()*100)
    OP_rw.append((SFE[name][MSFE[name][MSFE[name]==MSFE[name].min()].index].values.ravel()<SFE_rw.values).mean()*100)
    index.append(name)
OP_ar=pd.Series(OP_ar)
OP_ar.index=index
OP_rw=pd.Series(OP_rw)
OP_rw.index=index

OP_ar.plot.bar(rot=1)
plt.title('Percentage of better previsions than the AR(1)')
plt.savefig('Plots/OP_ar.png', dpi=500)

OP_rw.plot.bar(rot=1)
plt.title('Percentage of better previsions than the random walk')
plt.savefig('Plots/OP_rw.png', dpi=500)

#%% rolling variance
plt.figure(figsize=(10, 6))
plt.plot(y_std.index,y_std, linestyle='-', color = "black")
plt.title(f'{dep} standard deviation in each rolling window')
plt.xlabel('Time')
plt.ylabel('SD')
plt.grid(True)
plt.savefig('Plots/SD.png', dpi=500)
plt.show()

#%% dep change
d_y=y_true-y_true.shift(1) #this is also the forecast errors series of the random walk
plt.figure(figsize=(10, 6))
plt.plot(d_y.index,d_y, linestyle='-', color = "black")
plt.title(f'{dep} monthly change')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()

#correlation between the change from today and tomorrow and the predictors today
A=final[(final.index>=y_true.index.min()) & (final.index<y_true.index.max())]
A['d_y']=d_y.shift(-1)
A['y_ar']=y_ar
A['y_rw']=y_rw
corr_pred_d_y=A.corr()['d_y'].drop(['d_y','y_ar','y_rw'])  #since 
corr_ar_rw_d_y=A.corr()['d_y'].drop('d_y')[['y_ar','y_rw']]
