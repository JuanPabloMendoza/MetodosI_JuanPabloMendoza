import numpy as np
import matplotlib.pyplot as plt

import sympy as sym
from scipy.stats import chi2
import scipy.optimize as spo


#data = np.loadtxt('https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/MinimosLineal.txt')
data = np.loadtxt('https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/MinimosCuadratico.txt')

#data = np.loadtxt('https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/Exponencial.dat')

x = data[:,0]
y = data[:,1]
N = len(x)

sigma = np.random.normal(loc=0,scale=10,size=N)
sigma = np.abs(sigma)
print(sigma)

plt.errorbar(x,y,yerr=sigma,fmt='.',color='r')

# Definimos el modelo con sus parametros
def GetModel1(x,p):
    
    y = 0.
    for n in range(len(p)):
        y += p[n]*x**n
    
    return y

def GetModel2(x,p):
    
    A,B = p
    C  = 0.
    return A*np.exp(B*x)+C

def Chi2(p,x,y,sigma,Model):
    return np.sum( ( ( y - Model(x,p) )/sigma )**2  + np.log(sigma) ) 

#primer modelo
p0 = np.ones(3)
nll = lambda *p: Chi2(*p)

result = spo.minimize( nll, p0, args=(x,y,sigma,GetModel1), options={"disp":True},method='Nelder-mead')

if result.success:
    print('Success!')
    print(f"x={result.x} y = {result.fun}")
else:
    print('could not find a minimum')
    print(f"x={result.x} y = {result.fun}")
    
param = result.x

ObsChi2 = Chi2(param,x,y,sigma,GetModel1)

def Chi2Reducido(p,Model):
    return Chi2(p,x,y,sigma,Model)/(N-len(p))

Chi2R = Chi2Reducido(param,GetModel1)

#segundo modelo
p0 = np.ones(2)
nll = lambda *p: Chi2(*p)
result1 = spo.minimize( nll, p0, args=(x,y,sigma,GetModel2), options={"disp":True})

param1 = result1.x
ObsChi2 = Chi2(param1,x,y,sigma,GetModel2)

Chi2RE = Chi2Reducido(param1,GetModel2)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.errorbar(x,y,yerr=sigma,fmt='o',color='k',label='Data')

t = np.linspace(np.min(x),np.max(x),100)
ax.plot(t,GetModel1(t,param),lw=3,label='Polinomial')
ax.plot(t,GetModel2(t,param1),lw=3,label='Exponential')
ax.legend()

plt.show()

Chi2V = []
npfit = []

nparams = 15

for i in range(2,nparams):
    
    p0 = np.ones(i)
    result = spo.minimize( nll, p0, args=(x,y,sigma,GetModel1), options={"disp":True},method='Nelder-Mead')
    
    if result.success:
        
    
        param = result.x
    
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)
        ax.errorbar(x,y,yerr=sigma,fmt='o',color='r',label='Data')

        t = np.linspace(np.min(x),np.max(x),100)
        ax.plot(t,GetModel1(t,param),lw=3,label='Polinomio_{}'.format(i))
        ax.legend()
    
        plt.savefig('Ajuste_%.0f.jpg' %(i))
        plt.close()
    
        Chi2V.append( Chi2Reducido(param,GetModel1) )
        npfit.append(i)
        
plt.scatter(npfit,np.array(Chi2V),color='k',marker='^')
plt.axhline(y=Chi2RE,color='r')
plt.yscale('log')

plt.show()