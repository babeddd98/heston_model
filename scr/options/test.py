import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import newton
from scipy.stats import norm
from get_data import get_expiration_date

def black_scholes(S, K, r, q, T, sigma, option_type):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price

def implied_volatility(S, K, r, q, T, option_price, option_type, initial_guess=0.2):
    def func(sigma):
        return black_scholes(S, K, r, q, T, sigma, option_type) - option_price
    return newton(func, initial_guess, tol=1e-6)

ticker = "TSLA"
# Retrieve data for Tesla options from yfinance
tesla_options = yf.Ticker(ticker).option_chain()
tesla_calls = tesla_options.calls
tesla_puts = tesla_options.puts

# Combine data for calls and puts
tesla_options_data = pd.concat([tesla_calls, tesla_puts])

# Compute the implied volatility for each option
implied_vols = []
for index, row in tesla_options_data.iterrows():
    S = row['lastPrice']
    K = row['strike']
    r = 0.05
    q = 0.0
    T = (get_expiration_date(row['contractSymbol'],ticker) - pd.Timestamp.today()).days / 365
    option_price = row['lastPrice']
    option_type = 'call' if row['type'] == 'call' else 'put'
    initial_guess = 0.2
    implied_vol = implied_volatility(S, K, r, q, T, option_price, option_type, initial_guess)
    implied_vols.append(implied_vol)

tesla_options_data['implied_volatility'] = implied_vols

# Plot the volatility surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(tesla_options_data['strike'], tesla_options_data['maturity'], tesla_options_data['implied_volatility'] )
