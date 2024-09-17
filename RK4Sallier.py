# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:25:13 2024

@author: simon
"""


import numpy as np
import matplotlib.pyplot as plt

# Conditions initiales
S0 = 99995  # Nombre initial de personnes susceptibles
I0 = 5      # Nombre initial de personnes infectées
T0 = 0      # Nombre initial de personnes traitées
R0 = 0      # Nombre initial de personnes rétablies
N = S0 + I0 + R0 + T0  # Population totale

# Paramètres
beta = 1.5  # Taux moyen de rencontres par unité de temps
delta = 0.8  # Réduction de l'infectiosité suite au traitement
gamma = 0.2  # Taux de guérison
etha = 0.05  # Taux de mortalité des personnes traitées
alpha = 0.0  # Proportion d'individus traités par unité de temps
tf = 250  # Durée de la simulation
h = 0.5  # Pas de temps

# Définition des fonctions pour la méthode de Runge-Kutta 4

def K1_S(Sn,In,Tn):
    x=-(beta/N)*(In+delta*Tn)*Sn
    return x

def K1_I(Sn,In,Tn):
    x=(beta/N)*Sn*(In+delta*Tn)-(alpha+gamma)*In
    return x

def K1_T(Sn,In,Tn):
    x=alpha*In-etha*Tn
    return x

def K1_R(Sn,In,Tn):
    x=gamma*In+etha*Tn
    return x

def K2_S(Sn,In,Tn):
    x=-(beta/N)*((In+(h/2)*K1_I( Sn, In, Tn))+delta*(Tn+(h/2)*K1_T( Sn, In, Tn)))*(Sn+(h/2)*K1_S(Sn,In,Tn))
    return x

def K2_I(Sn,In,Tn):
    x=(beta/N)*(Sn+(h/2)*K1_S( Sn, In, Tn))*((In+(h/2)*K1_I( Sn, In, Tn))+delta*(Tn+(h/2)*K1_T( Sn, In, Tn)))-(alpha+gamma)*(In+(h/2)*K1_I( Sn, In, Tn))
    return x

def K2_T(Sn,In,Tn):
    x=alpha*(In+(h/2)*K1_I( Sn, In, Tn))-etha*(Tn+(h/2)*K1_T( Sn, In, Tn))
    return x

def K2_R(Sn,In,Tn):
    x=gamma*(In+(h/2)*K1_I( Sn, In, Tn))+etha*(Tn+(h/2)*K1_T( Sn, In, Tn))
    return x

def K3_S(Sn,In,Tn):
    x=-(beta/N)*((In+(h/2)*K2_I( Sn, In, Tn))+delta*(Tn+(h/2)*K2_T( Sn, In, Tn)))*(Sn+(h/2)*K2_S(Sn,In,Tn))
    return x

def K3_I(Sn,In,Tn):
    x=(beta/N)*(Sn+(h/2)*K2_S( Sn, In, Tn))*((In+(h/2)*K2_I( Sn, In, Tn))+delta*(Tn+(h/2)*K2_T( Sn, In, Tn)))-(alpha+gamma)*(In+(h/2)*K2_I( Sn, In, Tn))
    return x

def K3_T(Sn,In,Tn):
    x=alpha*(In+(h/2)*K2_I( Sn, In, Tn))-etha*(Tn+(h/2)*K2_T( Sn, In, Tn))
    return x

def K3_R(Sn,In,Tn):
    x=gamma*(In+(h/2)*K2_I( Sn, In, Tn))+etha*(Tn+(h/2)*K2_T( Sn, In, Tn))
    return x

def K4_S(Sn,In,Tn):
    x=-(beta/N)*((In+h*K3_I( Sn, In, Tn))+delta*(Tn+h*K3_T( Sn, In, Tn)))*(Sn+h*K3_S(Sn,In,Tn))
    return x

def K4_I(Sn,In,Tn):
    x=(beta/N)*(Sn+h*K3_S( Sn, In, Tn))*((In+h*K3_I( Sn, In, Tn))+delta*(Tn+h*K3_T( Sn, In, Tn)))-(alpha+gamma)*(In+h*K3_I( Sn, In, Tn))
    return x

def K4_T(Sn,In,Tn):
    x=alpha*(In+h*K3_I( Sn, In, Tn))-etha*(Tn+h*K3_T( Sn, In, Tn))
    return x

def K4_R(Sn,In,Tn):
    x=gamma*(In+h*K3_I( Sn, In, Tn))+etha*(Tn+h*K3_T( Sn, In, Tn))
    return x

# Implémentation de la Méthode RK4 dans des listes
s=[S0]
i=[I0]
t=[T0]
r=[R0]

for k in range(tf):
    S=s[k] + (h/6)*(K1_S(s[k],i[k],t[k])+2*K2_S(s[k],i[k],t[k])+2*K3_S(s[k],i[k],t[k])+K4_S(s[k],i[k],t[k]))
    I=i[k] + (h/6)*(K1_I(s[k],i[k],t[k])+2*K2_I(s[k],i[k],t[k])+2*K3_I(s[k],i[k],t[k])+K4_I(s[k],i[k],t[k]))
    T=t[k] + (h/6)*(K1_T(s[k],i[k],t[k])+2*K2_T(s[k],i[k],t[k])+2*K3_T(s[k],i[k],t[k])+K4_T(s[k],i[k],t[k]))
    R=r[k] + (h/6)*(K1_R(s[k],i[k],t[k])+2*K2_R(s[k],i[k],t[k])+2*K3_R(s[k],i[k],t[k])+K4_R(s[k],i[k],t[k]))
    s.append(S)
    i.append(I)
    t.append(T)
    r.append(R)
   
# Tracé des résultats
plt.plot(s, label='Susceptibles')
plt.plot(i, label='Infectés')
plt.plot(t, label='Traités')
plt.plot(r, label='Rétablis')
plt.xlabel('Temps')
plt.ylabel('Nombre de personnes')
plt.legend()
plt.show()