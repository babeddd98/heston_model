import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
# n=5000                # nombre de points simulés
# t=1                   # nombre de dimensions 
# sigma = np.sqrt(t/n)  # écart-type
# y=[0]
# y = sigma*np.random.randn(n) # vecteur de n valeurs aléatoires distribuées normalement
# y = np.cumsum(y)             # somme cumulative des yi

# x=[i for i in range(n)]      
# plt.plot(x, y, color='red', lw=1) 
# plt.xlabel("temps t")
# plt.grid(axis='y',linestyle='dotted', color='b')
# plt.show()


# GEOMETRIC BROWNIAN MOTION
sigmas = [0.2, 1.0, 3.0]
colors = ['red', 'green', 'blue']
mu = 10
n=5000                # nombre de points simulés
t=1                   # nombre de dimensions 
i = 0
for sigma in sigmas:
    dt_sqrt = np.sqrt(t/n)  # écart-type
    y=[0]
    y = sigma*dt_sqrt*np.random.randn(n) # vecteur de n valeurs aléatoires distribuées normalement
    y = np.cumsum(y)             # somme cumulative des yi
    x=[i for i in range(n)]
    x = np.array(x)
    y = mu*x*dt_sqrt**2 + x*y
    plt.plot(x, y, label=f"mu = {mu}; sigma = {sigma}", color=colors[i], lw=1) 
    i += 1

plt.xlabel("temps t")
plt.legend()
plt.grid(axis='y',linestyle='dotted', color='b', lw=1)
plt.show()
