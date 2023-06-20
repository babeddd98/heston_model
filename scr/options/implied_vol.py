from black_sholes import bs_call
from get_data import get_market_data
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import pandas as pd
import numpy as np
from scipy.interpolate import interp2d, Rbf
from matplotlib import cm
from algo_newton_raphson import vol_implicite
from mpl_toolkits.mplot3d import Axes3D

ticker = "^XSP"
r = 0.029 #risk-free rate
S0 = 403.78 # current spot price 

# sigma = 0.029
calls = get_market_data(ticker)
ttm = []  
vol = []
strike = []
for call in calls:
    for i in range(len(call)):
        c = call["Last Price"][i]
        K = call["Strike"][i]
        t = call["Time to Maturity"][i].days/365
        
        # def bs(sigma):
        #     return c - bs_call(S0,K,0,t,r,sigma)
        # vol_imp = brentq(bs, 0.001, 1.0)
        vol_imp = vol_implicite(S0,K,t,r,c)
        # vol_imp = call["Implied Volatility"][i]
        ttm.append(t)
        vol.append(vol_imp)
        strike.append(K)

f = interp2d(strike, ttm, vol, kind="cubic")
# f = Rbf(strike, ttm, vol, function='thin_plate', smooth=5, episilon=5)

d = {"Strike":strike, "TTM":ttm, "Vol":vol}
data = pd.DataFrame(d)
data.to_excel("vol_option.xlsx")

plot_strikes = np.linspace(data['Strike'].min(), data['Strike'].max(),25)
plot_ttm = np.linspace(0, data['TTM'].max(), 25)
ax = Axes3D(plt.figure())
X, Y = np.meshgrid(plot_strikes, plot_ttm)
Z = np.array([f(x,y) for xr, yr in zip(X, Y) for x, y in zip(xr,yr) ]).reshape(len(X), len(X[0]))
# Z = f(X,Y)
ax.plot_surface(X, Y, np.log10(Z), rstride=1, cstride=1, cmap='cool')
ax.set_xlabel('Strikes')
ax.set_ylabel('Time-to-Maturity')
ax.set_zlabel('Implied Volatility')

# ax.zaxis.set_scale('log')
plt.show()
# plt.scatter(calls[-1]["Strike"], calls[-1]["Implied Volatility"])
# plt.show()
# print(calls)