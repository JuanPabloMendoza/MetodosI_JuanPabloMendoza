import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sym
from scipy import integrate
from tqdm import tqdm
sym.init_printing(use_unicode=True)


def f(x):
   # return (x-1)**2
    return x**2 + 10*np.sin(x)
    #return x**4 - 2*x**2 -20*x -6
    
Dx = lambda f,x,h=1e-5: (f(x+h) - f(x-h))/(2*h)
_x = np.linspace(-10,10,50)
F = f(_x)

def Minimizer(f, N=300, gamma=0.01):
    
    r = np.zeros(N)
    # Seed
    r[0] = np.random.uniform(-5,5)
    
    for i in tqdm(range(1,N)):
        r[i] = r[i-1] - gamma*Dx(f,r[i-1])
        
    return r

x = Minimizer(f)

Cost = f(x)

fig = plt.figure(figsize=(7,2))
ax = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(1,2,2)

def init():
    
    ax.set_xlim(_x[0],_x[-1])
    
    ax.set_xlabel(r'$x$',fontsize=10)
    ax.set_ylabel(r'$f(x)$',fontsize=10)
    
    ax1.set_xlabel(r'$N$',fontsize=10)
    ax1.set_ylabel(r'$Cost \ Function$',fontsize=15)
    
def Update(i):
    
    ax.clear()
    init()
    ax.plot(_x,f(_x))
    ax.axhline(y=f(x[i]),ls='--',color='r')
    ax.scatter(x[i],f(x[i]), marker='o',color='r', label=r'$N=%.0f$'%(i),s=50)                 
    ax1.scatter(i,Cost[i],marker='.',color='k')                  
    
    ax.legend()


Animation = animation.FuncAnimation(fig, Update, frames=len(x),init_func=init)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800 )