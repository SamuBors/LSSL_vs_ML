#%% libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import statistics
from datetime import datetime
import statsmodels.api as sm
import os
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

#%% directory
os.chdir('/Users/samueleborsini/Library/Mobile Documents/com~apple~CloudDocs/Università/Economics and econometrics/II anno/Machine Learning/Project python')

#%% depedent variable
dep='INDPRO'

#%% data imoprting
raw_data=pd.read_csv('Data/current.csv') #dataset
raw_data['sasdate'][1:] = pd.to_datetime(raw_data['sasdate'][1:], format='%m/%d/%Y') #destringing the date

#%% transformations and dropping NaNs
data=raw_data.drop(['ACOGNO','ANDENOx','VIXCLSx','TWEXAFEGSMTHx','UMCSENTx','NONBORRES'], axis=1) #dropping problematic variables
transf=data.iloc[0][1:] #transformations row
data=data.iloc[1:] #data dataframe
dates=data['sasdate'].copy() #dates
data=data.drop('sasdate',axis=1) #data without the dates

transformations={
    1: lambda x: x,
    2: lambda x: x-x.shift(12),
    3: lambda x: x-x.shift(12) - (x.shift(12) - x.shift(24)),
    4: lambda x: np.log(x),
    5: lambda x: np.log(x)-np.log(x.shift(12)),
    6: lambda x: np.log(x)-np.log(x.shift(12))-(np.log(x.shift(12))-np.log(x.shift(24))),
    7: lambda x: (x/x.shift(12)-1)-(x.shift(12)/x.shift(24)-1)
    } #transformations function

data_transf=pd.DataFrame()
for column_name in data.columns:
    data_transf[column_name] = transformations[transf.loc[column_name]](data[column_name]) #applying the transformations

dates=dates[24:].reset_index(drop=True) #dropping the missing values
data_transf=data_transf.iloc[24:].reset_index(drop=True)

final=data_transf.copy()
final.index=dates #using the dates as index

final=final[final.index<pd.to_datetime('1/1/2020', format='%m/%d/%Y')]

#%% ADF tests
pval=pd.Series(np.nan,index=data_transf.columns)
X = sm.add_constant(range(0,final.shape[0]))
final_resid=pd.DataFrame()

for column in data_transf.columns:
    model = sm.OLS(final[column], X).fit() #computing the time trend and constant
    final_resid[column]=model.resid #taking the series without the trend and the constant
    result = adfuller(model.resid, maxlag=12, autolag='BIC') #ADF test, with at most 12 lags (it chooses the number of lags between 1 and 12 that gives the best BIC)
    pval.loc[column]=result[1] #storing the p-value of each test

#%% plots
name='M2SL'
plt.figure(figsize=(10, 6))
plt.plot(final[name], linestyle='-', color = "red")
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)

#%% exporting the final dataset
final.to_csv('Data/final.csv',index=True)

#%% correlation matrix plot
corr_matr=final.corr()
order=corr_matr['RPI'].sort_values(ascending=False).index
corr_matr=corr_matr.reindex(order, axis=0).reindex(order, axis=1)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matr, cmap='coolwarm', fmt=".2f", linewidths=0.00000001, xticklabels=False, yticklabels=False)
plt.title('Correlation Heatmap')
plt.savefig('Plots/correlation.png', dpi=500)
plt.show()

#%% NONBORRES
nonborres=raw_data['NONBORRES'][1:]
nonborres.index=raw_data['sasdate'][1:]
nonborres=transformations[raw_data['NONBORRES'].loc[0]](nonborres)
plt.figure(figsize=(10, 6))
plt.plot(nonborres.index, nonborres, linestyle='-', color = "red")
plt.title('NONBORRES')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.savefig('Plots/NONBORRES.png', dpi=500)



