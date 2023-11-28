import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
from scipy import integrate

def Likelihood(mu,n,b,s):
    
    l = mu*s + b
    
    L = np.exp( -l ) * l**n / np.math.factorial(int(n))
    
    return L


def JointLikelihood(mu,data):
    
    N = data.shape[0]
    n = data[:,0]
    b = data[:,1]
    s = data[:,2]
    
    JointL = 1.
    
    for c in range(N): 
        JointL *= Likelihood(mu, n[c], b[c], s[c])
    return JointL

def JointLogLikelihood(mu,data):
    
    N = data.shape[0]
    
    n = data[:,0]
    b = data[:,1]
    s = data[:,2]
    
    JointL = 0.
    
    for c in range(N):
        JointL += np.log(Likelihood(mu,n[c],b[c],s[c]))
        
    return JointL

#1 dato
n = np.array([0])
b = np.array([0])
s = np.array([1])

N = n.shape[0]

data = np.zeros((N,3))

data[:,0] = n
data[:,1] = b
data[:,2] = s

mu = np.linspace(0.,4.,100)
JointLike = JointLikelihood(mu,data)


mup = 0.
I = 0.
tolerancia = 1e-4

while np.abs(I - 0.95) > tolerancia:
    
    I = integrate.quad( JointLikelihood, 0.,mup, args=(data) )[0]
    
    mup += tolerancia
    
plt.plot(mu,JointLike)
plt.axvline(x=mup,color='r')
plt.show()


#1 dato intento 1
n = np.array([1])
b = np.array([0])
s = np.array([1])

N = n.shape[0]

data1 = np.zeros((N,3))

data1[:,0] = n
data1[:,1] = b
data1[:,2] = s

JointLike1 = JointLikelihood(mu,data1)

ii = np.where( JointLike1 == np.amax(JointLike1) )
muhat = mu[ii][0]

plt.plot(mu,JointLike1)
plt.axvline(x=mup,color='r')
plt.axvline(x=muhat,color='k')

plt.show()


#2 datos
n = np.array([1,0])
b = np.array([0,0])
s = np.array([1,1])

N = n.shape[0]

data2 = np.zeros((N,3))

data2[:,0] = n
data2[:,1] = b
data2[:,2] = s


Norm = integrate.quad( JointLikelihood, 0, np.inf, args=(data2) )[0]
JointLike2 = JointLikelihood(mu,data2)/Norm

ii = np.where( JointLike2 == np.amax(JointLike2) )
muhat = mu[ii][0]

plt.plot(mu,JointLike2)
plt.axvline(x=mup,color='r')
plt.axvline(x=muhat,color='k')

plt.show()


 




