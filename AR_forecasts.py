#%% libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import statistics
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import os
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#%% directory
os.chdir('/Users/samueleborsini/Library/Mobile Documents/com~apple~CloudDocs/Università/Economics and econometrics/II anno/Machine Learning/Project python')

#%% importing cleaned dataset
final=pd.read_csv('Data/final.csv')
final['sasdate']= [datetime.strptime(date_str, '%Y-%m-%d') for date_str in final['sasdate']]
final.index=final['sasdate']
final=final.drop('sasdate',axis=1)

#%% AR(1) forecasts
y=final[dep]
sm.graphics.tsa.plot_acf(y, lags=24, zero=False)
plt.show()
AR = AutoReg(y,lags=1).fit()
print(AR.summary())
y_ar=AR.predict()
y_ar.to_csv('Data/y_ar.csv',index=True)

#%% resiudals analysis
res=y-y_ar
plt.figure(figsize=(10, 6))
plt.plot(y_ar.index,res,linestyle='-', color = "deepskyblue", linewidth=0.9, label='residulas')
plt.title('Residuals')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.show()

sm.graphics.tsa.plot_acf(res[1:], lags=24, zero=False)
plt.show()

#%% AR forecasts
plt.figure(figsize=(10, 6))
plt.plot(y.index,y,linestyle='-', color = "black", label='True series')
plt.plot(y_ar.index,y_ar,linestyle='--', color = "deepskyblue", linewidth=0.9, label='AR(1)')
plt.title(fr'True series and AR(1) forecasts ($\hat\alpha$={round(AR.params[1],2)})')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.savefig('Plots/Forecasts/AR_forecasts.png', dpi=500)
plt.show()