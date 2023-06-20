# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:54:53 2019

@author: SUN
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ModuleBlackScholes import Fwd #Importation Fwd
from ModuleBlackScholes import bs_call #Importation bs_call
from ModuleBlackScholes import bs_put #Importation bs_put
from ModuleBlackScholes import delta_call #Importation delta_call
from ModuleBlackScholes import delta_put #Importation delta_put
from ModuleBlackScholes import bs_gamma #Importation bs_gamma


#---------------
# Forward EURUSD
#---------------

S = 1.135 # Spot EURUSD
print("EURUSD Spot:" "%.5f" % S)

r = 2.60/100 # Taux sans risque $
print("Taux sans risque $: ", "%.2f" % (r*100), "%")

q = -0.31/100 # Taux sans risque €
print("Taux sans risque €: ", "%.2f" % (q*100), "%")

T = 0.25 # 3 mois
print("Echéance: ", "%.2f" % T)

Fw = Fwd(S, r, q, T)

print("EURUSD Forward 3 mois:" "%.5f" % Fw)

#------------------
# Etude Call EURUSD
#------------------

S = 1.135 # Spot EURUSD

K = 1.1432
sigma = 6.16/100
T = 0.25 # 3 mois
r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

print("Volatilité: ", "%.2f" % (sigma*100), "%")

print("Strike:" "%.5f" % K)

prime_call = bs_call(S,K,sigma,T,r,q)

print("prime_call:" "%.5f" % prime_call)


# Variation Prime Call = F(S), K fixe
#------------------------------------

lb = 0.7
ub = 1.7

lb = 1.00
ub = 1.28

x=np.linspace(lb,ub,100)

Prime_V = [bs_call(xi,K,sigma,T,r,q) for xi in x]

Abcisse_V = [0 for xi in x]

plt.plot(x,Prime_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Prime call')
plt.xlabel("EURUSD S")
plt.show()
#plt.savefig('mon_graphe')

# Variation Prime Call = F(K), S fixe
#------------------------------------

S = 1.135 # Spot EURUSD

lb = 0.7
ub = 1.7

lb = 1.00
ub = 1.28

x=np.linspace(lb,ub,100)
Prime_V = [bs_call(S,xi,sigma,T,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Prime_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Prime call')
plt.xlabel("Strike K")
plt.show()

# Variation Prime Call = F(T), fonction de maturité
#--------------------------------------------------

# Cas At the money

S = 1.1432 # Spot EURUSD
print("EURUSD Spot:" "%.5f" % S)

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

v_bs_call = bs_call(S,K,sigma,T,r,q)

print("bs_call:" "%.5f" % v_bs_call)


lb = 0.01
ub = 2

x=np.linspace(lb,ub,100)
Prime_ATM_V = [bs_call(S,K,sigma,xi,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Prime_ATM_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Prime call ATM')
plt.xlabel("Maturité")
plt.show()
#plt.savefig('mon_graphe')

# Cas In the money

S = 1.175 # Spot EURUSD

print("EURUSD Spot:" "%.5f" % S)

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

v_bs_call = bs_call(S,K,sigma,T,r,q)

print("bs_call:" "%.5f" % v_bs_call)


lb = 0.01
ub = 2

x=np.linspace(lb,ub,100)
Prime_ITM_V = [bs_call(S,K,sigma,xi,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Prime_ITM_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Prime call ITM')
plt.xlabel("Maturité")
plt.show()

# Cas Out of the money

S = 1.115 # Spot EURUSD

print("EURUSD Spot:" "%.5f" % S)

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

v_bs_call = bs_call(S,K,sigma,T,r,q)

print("bs_call:" "%.5f" % v_bs_call)


lb = 0.01
ub = 2

x=np.linspace(lb,ub,100)
Prime_OTM_V = [bs_call(S,K,sigma,xi,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Prime_OTM_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Prime call OTM')
plt.xlabel("Maturité")
plt.show()

# Variation Prime Call = F(T), fonction de maturité, 3 cas
#---------------------------------------------------------

x=np.linspace(lb,ub,100)
plt.plot(x,Prime_ATM_V, 'k')
plt.plot(x,Prime_ITM_V, 'g')
plt.plot(x,Prime_OTM_V, 'r')
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Prime call')
plt.xlabel("Maturité")
plt.show()
#plt.savefig('mon_graphe')

#------------------
# Etude put EURUSD
#------------------

S = 1.135 # Spot EURUSD

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €
T = 0.25 # 3 mois

K = 1.1432
sigma = 6.16/100

prime_put = bs_put(S,K,sigma,T,r,q)

print("prime_put:" "%.5f" % prime_put)


# Variation Prime put = F(S), K fixe
#------------------------------------

lb = 0.7
ub = 1.7

lb = 1.00
ub = 1.28

x=np.linspace(lb,ub,100)
Prime_V = [bs_put(xi,K,sigma,T,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Prime_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Prime put')
plt.xlabel("EURUSD S")
plt.show()
#plt.savefig('mon_graphe')

# Variation Prime put = F(K), S fixe
#------------------------------------

S = 1.135 # Spot EURUSD

lb = 0.7
ub = 1.7

lb = 1.00
ub = 1.28

x=np.linspace(lb,ub,100)
Prime_V = [bs_put(S,xi,sigma,T,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Prime_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Prime put')
plt.xlabel("Strike K")
plt.show()

#------------------------
# Etude Delta Call EURUSD
#------------------------

S = 1.135 # Spot EURUSD

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

v_delta_call = delta_call(S,K,sigma,T,r,q)

print("Delta_call:" "%.5f" % v_delta_call)


# Variation Delta Call = F(S), K fixe
#------------------------------------

lb = 0.7
ub = 1.7

lb = 1.00
ub = 1.28

x=np.linspace(lb,ub,100)
Delta_V = [delta_call(xi,K,sigma,T,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Delta_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Delta call')
plt.xlabel("EURUSD S")
plt.show()
#plt.savefig('mon_graphe')

# Variation Delta Call = F(K), S fixe
#------------------------------------

S = 1.135 # Spot EURUSD

lb = 0.7
ub = 1.7

lb = 1.00
ub = 1.28

x=np.linspace(lb,ub,100)
Delta_V = [delta_call(S,xi,sigma,T,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Delta_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Delta call')
plt.xlabel("Strike K")
plt.show()

# Variation Delta Call = F(T), fonction de maturité
#--------------------------------------------------

# Cas At the money

S = 1.1432 # Spot EURUSD
print("EURUSD Spot:" "%.5f" % S)

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

v_delta_call = delta_call(S,K,sigma,T,r,q)

print("Delta_call:" "%.5f" % v_delta_call)


lb = 0.01
ub = 2

x=np.linspace(lb,ub,100)
Delta_ATM_V = [delta_call(S,K,sigma,xi,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Delta_ATM_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Delta call ATM')
plt.xlabel("Maturité")
plt.show()
#plt.savefig('mon_graphe')

# Cas In the money

S = 1.175 # Spot EURUSD

print("EURUSD Spot:" "%.5f" % S)

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

v_delta_call = delta_call(S,K,sigma,T,r,q)

print("Delta_call:" "%.5f" % v_delta_call)


lb = 0.01
ub = 2

x=np.linspace(lb,ub,100)
Delta_ITM_V = [delta_call(S,K,sigma,xi,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Delta_ITM_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Delta call ITM')
plt.xlabel("Maturité")
plt.show()

# Cas Out of the money

S = 1.115 # Spot EURUSD

print("EURUSD Spot:" "%.5f" % S)

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

v_delta_call = delta_call(S,K,sigma,T,r,q)

print("Delta_call:" "%.5f" % v_delta_call)


lb = 0.01
ub = 2

x=np.linspace(lb,ub,100)
Delta_OTM_V = [delta_call(S,K,sigma,xi,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Delta_OTM_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Delta call OTM')
plt.xlabel("Maturité")
plt.show()

# Variation Delta Call = F(T), fonction de maturité, 3 cas
#---------------------------------------------------------

x=np.linspace(lb,ub,100)
plt.plot(x,Delta_ATM_V, 'k')
plt.plot(x,Delta_ITM_V, 'g')
plt.plot(x,Delta_OTM_V, 'r')
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Delta call')
plt.xlabel("Maturité")
plt.show()
#plt.savefig('mon_graphe')


#------------------------
# Etude Delta put EURUSD
#------------------------

S = 1.135 # Spot EURUSD

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

v_delta_put = delta_put(S,K,sigma,T,r,q)

print("Delta_put:" "%.5f" % v_delta_put)


# Variation Delta put = F(S), K fixe
#------------------------------------

lb = 0.7
ub = 1.7

lb = 1.00
ub = 1.28

x=np.linspace(lb,ub,100)
Delta_V = [delta_put(xi,K,sigma,T,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Delta_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Delta put')
plt.xlabel("EURUSD S")
plt.show()
#plt.savefig('mon_graphe')

#------------------------
# Etude Gamma EURUSD
#------------------------

S = 1.135 # Spot EURUSD

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

v_gamma = bs_gamma(S,K,sigma,T,r,q)

print("bs_gamma:" "%.5f" % v_gamma)


# Variation Gamma = F(S), K fixe
#------------------------------------

lb = 0.7
ub = 1.7

lb = 1.00
ub = 1.28

x=np.linspace(lb,ub,100)
Gamma_V = [bs_gamma(xi,K,sigma,T,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Gamma_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Gamma')
plt.xlabel("EURUSD S")
plt.show()
#plt.savefig('mon_graphe')

# Variation Gamma = F(K), S fixe
#------------------------------------

S = 1.135 # Spot EURUSD

lb = 0.7
ub = 1.7

lb = 1.00
ub = 1.28

x=np.linspace(lb,ub,100)
Gamma_V = [bs_gamma(S,xi,sigma,T,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Gamma_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Gamma')
plt.xlabel("Strike K")
plt.show()
#plt.savefig('mon_graphe')

# Variation Gamma = F(delta)
#---------------------------

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

lb = 1.00
ub = 1.28

x=np.linspace(lb,ub,100)
Delta_V = [delta_call(xi,K,sigma,T,r,q) for xi in x]
Gamma_V = [bs_gamma(xi,K,sigma,T,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(Delta_V,Gamma_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Gamma')
plt.xlabel("Delta Call")
plt.show()
#plt.savefig('mon_graphe')


# Variation Gamma = F(T), fonction de maturité
#--------------------------------------------------

# Cas At the money (Forward)

S = 1.1432 # Spot EURUSD
print("EURUSD Spot:" "%.5f" % S)

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

v_gamma = bs_gamma(S,K,sigma,T,r,q)

print("Gamma_call:" "%.5f" % v_gamma)


lb = 0.01
ub = 2

x=np.linspace(lb,ub,100)
Gamma_ATM_V = [bs_gamma(S,K,sigma,xi,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Gamma_ATM_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Gamma ATM')
plt.xlabel("Maturité")
plt.show()

# Cas In the money

S = 1.175 # Spot EURUSD
print("EURUSD Spot:" "%.5f" % S)

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

v_gamma = bs_gamma(S,K,sigma,T,r,q)

print("Gamma_call:" "%.5f" % v_gamma)


lb = 0.01
ub = 2

x=np.linspace(lb,ub,100)
Gamma_ITM_V = [bs_gamma(S,K,sigma,xi,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Gamma_ITM_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Gamma ITM')
plt.xlabel("Maturité")
plt.show()

# Cas Out of the money

S = 1.115 # Spot EURUSD
print("EURUSD Spot:" "%.5f" % S)

r = 2.60/100 # Taux sans risque $
q = -0.31/100 # Taux sans risque €

T = 0.25 # 3 mois

K = 1.1432

sigma = 6.16/100

v_gamma = bs_gamma(S,K,sigma,T,r,q)

print("Gamma_call:" "%.5f" % v_gamma)


lb = 0.01
ub = 2

x=np.linspace(lb,ub,100)
Gamma_OTM_V = [bs_gamma(S,K,sigma,xi,r,q) for xi in x]
Abcisse_V = [0 for xi in x]

x=np.linspace(lb,ub,100)
plt.plot(x,Gamma_OTM_V)
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Gamma OTM')
plt.xlabel("Maturité")
plt.show()

# Variation Gamma = F(T), fonction de maturité, 3 cas
#---------------------------------------------------------

x=np.linspace(lb,ub,100)
plt.plot(x,Gamma_ATM_V, 'k')
plt.plot(x,Gamma_ITM_V, 'g')
plt.plot(x,Gamma_OTM_V, 'r')
#plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Gamma')
plt.xlabel("Maturité")
plt.show()
#plt.savefig('mon_graphe')




