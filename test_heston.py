import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Define the Heston model


def heston_model(params, S0, K, r, T, market_price):
    kappa, theta, sigma, rho, v0 = params
    dt = 1/252
    n = int(T/dt)
    s = np.zeros(n)
    v = np.zeros(n)
    s[0] = S0
    v[0] = v0
    for i in range(1, n):
        dz1 = norm.rvs()
        dz2 = rho*dz1 + np.sqrt(1-rho**2)*norm.rvs()
        v[i] = abs(v[i-1] + kappa*(theta-v[i-1]) *
                   dt + sigma*np.sqrt(v[i-1]*dt)*dz1)
        s[i] = s[i-1] * np.exp((r - 0.5*v[i])*dt + np.sqrt(v[i]*dt)*dz2)
    model_price = np.maximum(s[-1] - K, 0)
    error = np.mean(np.square(model_price - market_price))
    return error


# Define the market data
S0 = 100
K = 110
r = 0.05
T = 1
market_price = 10

# Set initial parameter values
params0 = [0.5, 0.05, 0.2, -0.5, 0.05]

# Minimize the error function
result = minimize(heston_model, params0, args=(S0, K, r, T, market_price))

# Print the calibrated parameters
print(result.x)


# Set the calibrated parameters
kappa, theta, sigma, rho, v0 = result.x

# Define the Heston model


def heston_implied_vol(params, S0, K, r, T):
    kappa, theta, sigma, rho, v0 = params
    dt = 1/252
    n = int(T/dt)
    s = np.zeros(n)
    v = np.zeros(n)
    s[0] = S0
    v[0] = v0
    for i in range(1, n):
        dz1 = norm.rvs()
        dz2 = rho*dz1 + np.sqrt(1-rho**2)*norm.rvs()
        v[i] = abs(v[i-1] + kappa*(theta-v[i-1]) *
                   dt + sigma*np.sqrt(v[i-1]*dt)*dz1)
        s[i] = s[i-1] * np.exp((r - 0.5*v[i])*dt + np.sqrt(v[i]*dt)*dz2)
    model_price = np.maximum(s[-1] - K, 0)
    return model_price


# Define the strike prices to plot
strikes = np.linspace(80, 120, 41)

# Calculate the implied volatilities
implied_vols = []
for K in strikes:
    def f(sigma):
        params = [kappa, theta, sigma, rho, v0]
        model_price = heston_implied_vol(params, S0, K, r, T)
        return abs(norm.cdf(-np.log(model_price/S0)/sigma/np.sqrt(T)) - norm.cdf(-np.log(model_price/S0)/sigma/np.sqrt(T) + sigma*np.sqrt(T)))
    result = minimize(f, sigma)
    implied_vols.append(result.x[0])

# Plot the volatility smile
plt.plot(strikes, implied_vols, 'bo')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Heston Model Volatility Smile')
plt.show()
