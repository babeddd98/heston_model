import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
N = norm.cdf

def bs_call(S, K, t, T, r, sigma) -> float:
    d1 = (1/(sigma*np.sqrt(T-t)))*(np.log(S/K) + (r + 0.5*sigma**2)*(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    return S*N(d1) - K*np.exp(-r*(T-t))*N(d2)

def bs_put(S, K, t, T, r, sigma) -> float:
    d1 = (1/(sigma*np.sqrt(T-t)))*(np.log(S/K) + (r + 0.5*sigma**2)*(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    return -S*N(-d1) + K*np.exp(-r*(T-t))*N(-d2)

def long_call(S, K, bs_premium) -> np.ndarray:
    return np.maximum((S-K),0) - bs_premium

def long_put(S, K, bs_premium) -> np.ndarray:
    return np.maximum((K-S),0) - bs_premium

def short_call(S, K, bs_premium) -> np.ndarray:
    return np.array([-i for i in long_call(S,K,bs_premium)])

def short_put(S,K, bs_premium) -> np.ndarray:
    return np.array([-i for i in long_put(S,K,bs_premium)])

def plot_payoff(
        S: np.ndarray, 
        K: float,
        T: float,
        r: float,
        sigma: float, 
        pos="long", 
        option="call")->None:
    """Plots the payoff of a position on an option

    Args:
        S (np.ndarray): Spot price
        K (float): Strike price
        T (float): Maturity
        r (float): risk free rate
        sigma (float): implied volatility
        pos (str, optional): Position type. Defaults to "long".
        option (str, optional): Option type. Defaults to "call".
    """
    if option=="call":
        if pos=="long":
            payoff = long_call(S, K, bs_call(K, K, 0, T, r, sigma))
            current_value = [St - bs_call(St, K, 0, T, r, sigma) for St in S]
        elif pos=="short":
            payoff = short_call(S, K, bs_call(K, K, 0, T, r, sigma))
            current_value = [-bs_call(St, K, 0, T, r, sigma) + St for St in S]
        else:
            ValueError("Position must be long or short.")
    elif option=="put":
        if pos=="long":
            payoff = long_put(S, K, bs_put(K, K, 0, T, r, sigma))
            current_value = [bs_put(St, K, 0, T, r, sigma) - St for St in S]
        elif pos=="short":
            payoff = short_put(S, K, bs_put(K, K, 0, T, r, sigma))
            current_value = [-bs_put(St, K, 0, T, r, sigma) + St for St in S]
        else:
            ValueError("Position must be long or short.")
    else:
        ValueError("Option must be call or put.")

    plt.plot(S, payoff, label="payoff at maturity", color="blue")
    plt.plot(S, current_value, label=f"current value of the {option}", color="red")
    plt.grid(axis='both', linestyle='dotted', lw=0.5)
    plt.xlabel("Spot price $S_t$")
    plt.ylabel("Present value")
    plt.legend()
    plt.title(f"{pos} {option}")
    # plt.show()

T = 1.0          # Maturity
r = 0.025        # risk free rate
sigma = 0.2      # implied volatility
ticker = "^FCHI" # CAC40
data = yf.download(ticker, datetime(2023,1,1), datetime(2023,1,10))
K = float(data.head(1)["Close"])

S = np.linspace(0, 2*K, 150)

fig,ax = plt.subplots(nrows=2, ncols=2)

plt.subplot(221)
plot_payoff(S,K,T,r,sigma)
plt.subplot(222)
plot_payoff(S,K,T,r,sigma, "short")
plt.subplot(223)
plot_payoff(S,K,T,r,sigma, "long", "put")
plt.subplot(224)
plot_payoff(S,K,T,r,sigma, "short", "put")

plt.show()

# https://clinthoward.github.io/portfolio/2017/04/16/BlackScholesGreeks/