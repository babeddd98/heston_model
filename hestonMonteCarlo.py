import numpy as np

def heston_monte_carlo(
    S0, T, mu, kappa, theta, v0, rho, xi, steps, nsimu
    ):
    
    dt = T/steps
    size = (nsimu, steps)
    prices = np.zeros(size)
    St = S0 
    vt = v0
    for t in range(steps):
        wiener = np.random.multivariate_normal(np.array([0, 0]),
                                           cov=np.array([[1, rho],
                                                         [rho, 1]]),
                                           size=nsimu) * np.sqrt(dt)

        St = St + mu*St*dt + St*np.sqrt(vt) * wiener[:, 0]
        vt = vt + kappa*(theta-vt)*dt + xi*np.sqrt(vt)*wiener[:,1]
        prices[:,t] = St
        
    return prices