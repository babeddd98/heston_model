import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def d1(S, K, T, r, sigma):
    return (1/(sigma*np.sqrt(T)))*(np.log(S/K) + (r + sigma**2/2)*T)

def d2(S, K, T, r, sigma):
    return (1/(sigma*np.sqrt(T)))*(np.log(S/K) + (r - sigma**2/2)*T)

def call(S, K, T, d_1, d_2, r):
    return S*stats.norm.cdf(d_1) - K*np.exp(-r*T)*stats.norm.cdf(d_2)