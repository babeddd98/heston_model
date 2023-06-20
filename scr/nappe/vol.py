import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime
from scipy.interpolate import griddata


# Define the symbol of the underlying asset
symbol = "AAPL"  # Replace with the desired stock symbol
#aapl
expi = ['2023-05-12', 
        '2023-05-19', 
        '2023-05-26', 
        '2023-06-02', 
        '2023-06-09', 
        '2023-06-16', 
        '2023-06-23', 
        '2023-07-21', 
        '2023-08-18', 
        '2023-09-15', 
        '2023-10-20', 
        '2023-11-17', 
        '2023-12-15', 
        '2024-01-19', 
        '2024-03-15', 
        '2024-06-21', 
        '2024-09-20', 
        '2024-12-20', 
        '2025-01-17', 
        '2025-06-20', 
        '2025-12-19']

#AMZN
# expi = ['2023-05-12', '2023-05-19', '2023-05-26', '2023-06-02', '2023-06-09', '2023-06-16', '2023-06-23', '2023-07-21', '2023-08-18', '2023-09-15', '2023-10-20', '2024-01-19', '2024-03-15', '2024-06-21', '2024-09-20', '2025-01-17', '2025-06-20', '2025-12-19']

opt = []
for i in expi:
    # Retrieve the option chain data' using yfinance
    stock = yf.Ticker(symbol)
    op = stock.option_chain(i)
    opt.append(op.calls)

option_chain = pd.concat(opt)
# Access the strike prices and expiration dates from the option chain
strikes = option_chain.strike.values
expirations = pd.DataFrame()

expirations["year"] = "20" + option_chain.contractSymbol.str[4:6]


expirations["month"] = option_chain.contractSymbol.str[6:8]
expirations["day"] = option_chain.contractSymbol.str[8:10]

expirations["date"] = pd.to_datetime(dict(year=expirations.year, month=expirations.month, day=expirations.day))
expirations["date"] = expirations["date"].values.astype(np.int64) // 10**9

# Create a mesh grid for strikes and expirations
X, Y = np.meshgrid(expirations["date"], strikes)
l = option_chain.impliedVolatility.values
k = []

for i in range(len(expirations)): k.append(l)
# Access the implied volatility values from the option chain
# iv_values = option_chain.impliedVolatility.values.reshape(len(strikes), len(expirations))
iv_values = np.array(k)

extrapolated_grid = griddata((expirations['date'], strikes), iv_values[0], (X, Y), method='linear')

# Plot the volatility surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, extrapolated_grid, cmap='viridis')
ax.set_xlabel('Expiration')
ax.set_ylabel('Strike')
ax.set_zlabel('Implied Volatility')
ax.set_title(f"Volatility Surface {symbol}")
plt.savefig(f"vol_{symbol}.png")
# plt.show()
