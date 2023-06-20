import numpy as np
import matplotlib.pyplot as plt


rho = 2
n = 1000
mu = np.array([0,0])
cov = np.array([[1, rho] , [rho , 1]]) # matrice de covariance
w = np.random.multivariate_normal(mu, cov, size=n)
w = w.cumsum(axis=0)

plt.plot(w.T[0], color="blue", label="")
plt.plot(w.T[1], color="red", label="")
plt.title("Variables aléatoires corrélées")
plt.grid(axis='y',linestyle='dotted', color='b', lw=1)
plt.show()