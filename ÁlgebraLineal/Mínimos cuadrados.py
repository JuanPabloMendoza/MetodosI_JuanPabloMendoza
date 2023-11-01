import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

from scipy.stats import norm
from scipy.stats import t
from scipy.stats import chi2

import os
import os.path as path
import urllib.request


data = np.loadtxt('/Users/juanpablomendozaarias/Library/Mobile Documents/com~apple~CloudDocs/Universidad/Tercer Semestre/Métodos Computacionales/MetodosI_JuanPabloMendoza/ÁlgebraLineal/MinimosLineal.txt', encoding='utf-8')
#data = np.loadtxt('/Users/juanpablomendozaarias/Library/Mobile Documents/com~apple~CloudDocs/Universidad/Tercer Semestre/Métodos Computacionales/MetodosI_JuanPabloMendoza/ÁlgebraLineal/Minimos2.txt', encoding='utf-8')

x = data[:,0]
y = data[:,1]

N = len(x)
sigma = np.random.uniform(0,2.,size=N)
sigma
#plt.scatter(x,y)
plt.errorbar(x,y,yerr=sigma,fmt='o',color='k',label='Data')
plt.legend(loc=0)
#plt.show()

def GetFit(x,y,n=1):
    
    l = x.shape[0]
    b = y
    
    A = np.ones((l,n+1))
    
    for i in range(1,n+1):
        A[:,i] = x**i
        
    AT = np.dot(A.T,A)
    bT = np.dot(A.T,b)

    xsol = np.linalg.solve(AT,bT)
    
    return xsol


n = 1
param = GetFit(x,y,n)



def GetModel(x,p):
    
    y = 0.
    for i in range(len(p)):
        y += p[i]*x**i
        
    return y

X = sym.Symbol('x',real=True)
GetModel(X,param)
print(GetModel(X,param))

_x = np.linspace(np.min(x),np.max(x),50)
_y = GetModel(_x,param)

plt.errorbar(x,y,yerr=sigma,fmt='o',color='k',label='Data')
plt.plot(_x,_y,color='r',lw=2,label='Model')
plt.legend()
plt.show()


def GetError(x,y, p):
    l = x.shape[0]
    
    A = np.ones((l,n+1))
    
    for i in range(1,n+1):
        A[:,i] = x**i
    
    #Residuos
    R = y - np.dot(A,p)
    
    sigma2 = np.dot(R.T, R)/(len(y)-len(p))
    
    Cov = sigma2*np.linalg.inv(np.dot(A.T, A))
        
    print(sigma2)
    return Cov

Cov = GetError(x,y, param)
print(Cov)
print(param)


#print(f'Pendiente: {param[1]} +- {np.sqrt(Cov[1,1])}')
#print(f'a3: {param[2]} +- {np.sqrt(Cov[2,2])}')
print(f'ult. coff: {param[n]} +- {np.sqrt(Cov[n,n])}')
t_observado = (0.-param[n])/np.sqrt(Cov[n,n])
print(t_observado)

df = N - len(param) - 1
tcritico = t.ppf(0.975, df=df)
print(tcritico)
Descartar = False
if tcritico<t_observado:
    Descartar = True
print(f'Descartar: {Descartar}')

