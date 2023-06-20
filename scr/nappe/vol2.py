import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the symbol of the underlying asset
symbol = "AAPL"  # Replace with the desired stock symbol

# Retrieve the option chain data using yfinance
stock = yf.Ticker(symbol)
option_chain = stock.option_chain()

# Access the strike prices and expiration dates from the option chain
strikes = option_chain.calls['strike'].values
expirations = option_chain.expirationDates.values.astype(np.int64) // 10**9

# Create a mesh grid for strikes and expirations
X, Y = np.meshgrid(expirations, strikes)

# Access the implied volatility values from the option chain
iv_values = option_chain.calls['impliedVolatility'].values.reshape(len(strikes), len(expirations))

# Plot the volatility surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, iv_values, cmap='viridis')
ax.set_xlabel('Expiration')
ax.set_ylabel('Strike')
ax.set_zlabel('Implied Volatility')
ax.set_title('Volatility Surface')
plt.show()
