from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sym
from scipy import integrate
from tqdm import tqdm
sym.init_printing(use_unicode=True)

def funcion(x,y):
    return x**4 + y**4 - 2*(x-y)**2

def Gradient(f, x, y, h=1e-6):
    return (f(x+h)-f(x-h))/(2*h)

Dx = lambda x,y,h=1e-5: (funcion(x+h,y)-funcion(x-h,y))/(2*h)
Dy = lambda x,y,h=1e-5: (funcion(x,y+h)-funcion(x,y-h))/(2*h)

D = lambda x,y: np.array([Dx(x,y),Dy(x,y)])


def Minimizer(f, N = 200, gamma = 0.01):
    
    r = np.zeros((N,2))
    r[0] = np.random.uniform(-3,3,size=2)
    
    for i in range(1,N):
        r[i] = r[i-1] - gamma*D(r[i-1,0],r[i-1,1])
    
    return r
    
x = Minimizer(funcion)
print(x)
""" 
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

def init():
    
    ax.set_xlim(-10,10)
    #ax.set_ylim(0,3)
    
def Update(i):
    
    plot = ax.clear()
    init()
    plot = ax.plot(l,F)
    plot = ax.scatter(x[i],f(x[i]),color='r')
    
    return plot

Animation = animation.FuncAnimation(fig, Update, frames = len(x), init_func=init) """
