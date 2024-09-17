# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:40:10 2024

@author: simon
"""

import numpy as np
import matplotlib.pyplot as plt





#Parametres du modele
beta=8
alpha=0.8
gamma=0.5
eta=0.7
delta=0.8

#Nombre moyen de nouvelles infections dues à un individu malade
coeffR0 = (beta/(alpha+gamma)) + (alpha*delta*beta)/((alpha+gamma)*eta)



#Paramètres de temps
t0 = 0
tf = 400
h = 0.05


#Conditions initiales
S0=93
I0=2.0
T0=0.0
R0= 0.0
N = S0+I0 + R0  
valeurs = np.array([[S0,I0,T0,R0]])

#Euler explicite pour calculer les valeurs de la première itération.

S1 = S0 + h*(-beta*(I0 + delta*T0)/N)*S0
I1 = I0 + h*(beta*S0*(I0+delta*T0)/N - (alpha+gamma)*I0 )
T1 = T0 + h*(alpha*I0 - eta*T0)
R1 = R0 + h*(gamma*I0 + eta*T0)
valeurs = np.vstack((valeurs,[S1,I1,T1,R1]))





#Création de la fonction phi utilisée dans l'itération de newton
def phi(S0,I0,T0,R0,S1,I1,T1,R1):
    phi = np.array([S0/(1+h*beta*(I1+delta*T1)/(N))-S1,
                   (I0+(h*beta*delta*T1*S1)/(N))/(1+h*(alpha+gamma)-(h*beta*S1)/(N))-I1,
                   (T0+h*I1)/(1+h*eta)-T1,
                   R0+h*(gamma*I1+eta*T1)-R1])
    return phi 




#Création de la fonction Jacobienne utilisée dans les itérations de Newton.
def Jac(S0,I0,T0,R0,S1,I1,T1,R1):
    J = np.zeros((4,4))
    J[0,0]= -1
    J[1,1]= -1
    J[2,2]= -1
    J[3,3]= -1
    J[0,1] = (-S0*h*beta/(N))/(1+h*beta*(I1+delta*T1)/(N))**2
    J[0,2] = (-S0*h*beta*delta/(N))/(1+h*beta*(I1+delta*T1)/(N))**2
    J[1,0] = ((h*beta*delta*T1/(N))*(1+h*(alpha+gamma)-(h*beta/(N))*S1)+
              (I0+h*beta*delta*T1*S1/(N))*h*beta/(N))/((1+h*(alpha+gamma)-(h*beta/(N))*S1))**2
    J[1,2] = (h*beta*S1*delta/(N))/(1+h*(alpha+gamma)-(h*beta/(N))*S1)
    J[2,1] = h/(1+h*eta)
    J[3,1] = h*gamma
    J[3,2] = h*eta
    return J




#boucle avec condition d'arret qui va nous permettre de calculer les itérations de Newton ainsi que d'ajouter les valeurs calculées dans un tableau
i=2
while ( valeurs[i-1,0] > 0.05 or valeurs[i-1,1] > 0.05 and i<5000 ):
    N = valeurs[i-1,0] + valeurs[i-1,1] + valeurs[i-1,3] # car les patients traités ne peuvent plus être infecté
    temp = np.zeros((1,4))
    temp = (valeurs[i-1,:] - np.dot(np.linalg.inv(Jac(valeurs[i-2,0],valeurs[i-2,1],valeurs[i-2,2],valeurs[i-2,3], 
            valeurs[i-1,0], valeurs[i-1,1], valeurs[i-1,2], valeurs[i-1,3])),phi(valeurs[i-2,0],valeurs[i-2,1],valeurs[i-2,2],valeurs[i-2,3], 
            valeurs[i-1,0], valeurs[i-1,1], valeurs[i-1,2], valeurs[i-1,3]))).reshape(1,4)
    valeurs = np.vstack((valeurs,temp))
    i = i +1

#création du graphique
abscisse = np.arange(0, i)
plt.plot(abscisse, valeurs[:,0], label='succeptible')
plt.plot(abscisse, valeurs[:,1], label='infecté')
plt.plot(abscisse, valeurs[:,2], label='traité')
plt.plot(abscisse, valeurs[:,3], label='rétabli')
plt.legend()


   


