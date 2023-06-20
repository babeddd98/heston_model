# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:13:07 2019

@author: SUN
"""

#############################
# Stratégie Forward et Option
#############################

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

#---------------------------------------------------
# Payoff of long, short Forward & Option Call et Put
#---------------------------------------------------

# Fonction Payoff Long Forward
##############################

def Payoff_long_forward(S,Fw): # S: Spot, Fw: Forward
    Payoff_lg_fwd=S-Fw
    return(Payoff_lg_fwd)
  
S=1.1350
Fw=1.1400

Ex_Payoff_lg_fwd = Payoff_long_forward(S, Fw)

print("%.4f" % Ex_Payoff_lg_fwd)

# Courbe Payoff

x=np.linspace(0.5,2.0,100)
Payoff_V = [Payoff_long_forward(xi, Fw) for xi in x]
Abcisse_V = [0 for xi in x]

plt.plot(x,Payoff_V)
plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Payoff Long Fwd')
plt.xlabel("EURUSD")
plt.show()
#plt.savefig('mon_graphe')

# Fonction Payoff Short Forward
###############################

def Payoff_short_forward(S,FW):
    Payoff_sh_fwd=FW-S
    return(Payoff_sh_fwd)

S=1.1350
Fw=1.1400

Ex_Payoff_sh_fwd=Payoff_short_forward(S, Fw)

print("%.4f" % Ex_Payoff_sh_fwd)

# Courbe Payoff

x=np.linspace(0.5,2.0,100)
Payoff_V = [Payoff_short_forward(xi, Fw) for xi in x]
Abcisse_V = [0 for xi in x]

plt.plot(x,Payoff_V)
plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Payoff Short Fwd')
plt.xlabel("EURUSD")
plt.show()
#plt.savefig('mon_graphe')

# Fonction Payoff Option Call
#############################

def Payoff_long_call(S,K,P): # S: Spot, K: Strike P: Prime du Call
    if (S>=K):
      d_sk = S-K
    else:
        d_sk = 0 
    Payoff_lg_call=d_sk-P
    return(Payoff_lg_call)

S=1.1350
K=1.1378

P=0.0076

Ex_Payoff_lg_call=Payoff_long_call(S, K, P)

print("%.4f" % Ex_Payoff_lg_call)

# Courbe Payoff Option Call

x=np.linspace(1.1,1.19,100)
Payoff_V = [Payoff_long_call(xi, K, P) for xi in x]
Abcisse_V = [0 for xi in x]

plt.plot(x,Payoff_V)
plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Payoff Long Call')
plt.xlabel("EURUSD")
plt.show()
#plt.savefig('mon_graphe')


# Fonction Payoff Option Put
#############################

def Payoff_long_put(S,K,P):
    if (S<=K):
        d_sk = K-S
    else:
        d_sk = 0
    Payoff_lg_put=d_sk-P
    return(Payoff_lg_put)

S=1.1350
K=1.1378

P=0.0078

Ex_Payoff_lg_put = Payoff_long_put(S, K, P)

print("%.4f" % Ex_Payoff_lg_put)

# Courbe Payoff Option Put

x=np.linspace(1.1,1.19,100)
Payoff_V = [Payoff_long_put(xi, K, P) for xi in x]
Abcisse_V = [0 for xi in x]

plt.plot(x,Payoff_V)
plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Payoff Long Put')
plt.xlabel("EURUSD")
plt.show()
#plt.savefig('mon_graphe')

#----------------
# Option Strategy
#----------------


# Straddle
# Achat Call et Put, même Strike, même échéance
###############################################

# Fonction Straddle
#------------------

def Payoff_straddle(S,K,Pc,Pp):
    Payoff_straddle=(Payoff_long_call(S,K,Pc)+Payoff_long_put(S,K,Pp))
    return(Payoff_straddle)

#Straddle Delta 50 (Delta Call = 50, Delta Put = -50)

K=1.1378
Pc=0.0076
Pp=0.0078

S=1.1350

Ex_Payoff_straddle = Payoff_straddle(S,K,Pc,Pp)

print("%.4f" % Ex_Payoff_straddle)

# Courbe Payoff Stradle

x=np.linspace(1.1,1.175,100)
Payoff_straddle_V = [Payoff_straddle(xi,K,Pc,Pp) for xi in x]
Abcisse_V = [0 for xi in x]

plt.plot(x,Payoff_straddle_V)
plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Payoff Straddle')
plt.xlabel("EURUSD")
plt.show()
#plt.savefig('mon_graphe.pdf')




# Bull Spread (avec Call)
# Achat Call K1, Vente Call K2, K2 > K1, même échéance
# Bilan : 
#   S<K1, Payoff = -Delta_P; Delta_P = (P1 - P2) > 0
#   K1<=S<K2, Payoff = (S - K1) - Delta_P
#   S = K1 + Delta_P; Payoff = 0 
#   S>=K2, Payoff = (K2-K1) - Delta_P 
######################################################

# Fonction Bull Spread (avec call)
#---------------------------------

def Payoff_bull_spread_call(S,K1,P1,K2,P2):
    Payoff_bull_spread=Payoff_long_call(S,K1,P1)-Payoff_long_call(S,K2,P2)
    return(Payoff_bull_spread)

# Achat Call 1 : Delta 75, Vente Call 2 Delta 25 

K1=1.1242
P1=0.0164

K2=1.1508
P2=0.0028

S=1.1350

Ex_Payoff_bull_spread = Payoff_bull_spread_call(S,K1,P1,K2,P2)

print("%.4f" % Ex_Payoff_bull_spread)

# Courbe Payoff bull_spread

x=np.linspace(1.1,1.175,100)
Payoff_bull_spread_V = [Payoff_bull_spread_call(xi,K1,P1,K2,P2) for xi in x]
Abcisse_V = [0 for xi in x]

plt.plot(x,Payoff_bull_spread_V)
plt.plot(x,Abcisse_V, 'k')
plt.ylabel('Payoff Bull_spread_call')
plt.xlabel("EURUSD")
plt.show()
#plt.savefig('mon_graphe.pdf')


