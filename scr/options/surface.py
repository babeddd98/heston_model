import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import yfinance as yf
from datetime import datetime
from bs_model import BSM_call_value
data = yf.download("AAPL", start=datetime(2020,1,1), end=datetime(2023,1,1))
data = data.dropna()
dtypes = {'Date': 'str', 'Strike': 'float', 'Call_Bid': 'float', 'Call_Ask': 'float',
          'Call':'float','Maturity':'str','Put_Bid':'float','Put_Ask':'float','Put':'float'}
parse_dates = ['Date', 'Maturity']

data['BS_Imp_Vol'] = 0.0
data['TTM'] = 0.0
r = 0.0248 #risk-free rate
S0 = 153.97 # spot price as of 5/11/2017
div = 0.0182
for i in data.index:
    t = data['Date'][i]
    T = data['Maturity'][i]
    K = data['Strike'][i]
    Put = data['Put'][i]
    time_to_maturity = (T - t).days/365.0
    data.loc[i, 'TTM'] = time_to_maturity
    def func_BS(sigma):
        return BSM_call_value(S0, K, 0, time_to_maturity, r, sigma) - Put

    bs_imp_vol = brentq(func_BS, 0.03, 1.0)
    data.loc[i,'BS_Imp_Vol'] = bs_imp_vol

ttm = data['TTM'].tolist()
strikes = data['Strike'].tolist()
imp_vol = data['BS_Imp_Vol'].tolist()
f = interp2d(strikes,ttm,imp_vol, kind='cubic')

plot_strikes = np.linspace(data['Strike'].min(), data['Strike'].max(),25)
plot_ttm = np.linspace(0, data['TTM'].max(), 25)
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(plot_strikes, plot_ttm)
Z = np.array([f(x,y) for xr, yr in zip(X, Y) for x, y in zip(xr,yr) ]).reshape(len(X), len(X[0]))
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0.1)
ax.set_xlabel('Strikes')
ax.set_ylabel('Time-to-Maturity')
ax.set_zlabel('Implied Volatility')
fig.colorbar(surf, shrink=0.5, aspect=5)