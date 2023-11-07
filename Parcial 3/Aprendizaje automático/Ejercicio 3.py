import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sym
from scipy import integrate
from tqdm import tqdm
sym.init_printing(use_unicode=True)


def f(x, a=2):
   if x>a:
       return x-a
   else:
       return -x+a
    
Dx = lambda f,x,h=1e-5: (f(x+h) - f(x-h))/(2*h)

""" def Dx(x, a=2):
    if x>a:
        return 1
    else:
        return -1 """
    
N=100
_x = np.linspace(1,3,N)

    
def Minimizer(f, seed, N=300, gamma=0.01):
    
    r = np.zeros(N)
    # Seed
    r[0] = seed
    Grad = np.zeros(N)
    
    for i in tqdm(range(1,N)):
        r[i] = r[i-1] - gamma*Dx(f,r[i-1])
        Grad[i] = Dx(f,r[i-1])
        
    return r, Grad

seed = 2.2
x ,Grad = Minimizer(f,seed,N)
print(x)

F = np.zeros((N))
Cost = np.zeros((N))
for i in range(N):
    F[i] = f(_x[i])
    Cost[i] = f(x[i])


    

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
    ax.plot(_x,F)
    ax.axhline(y=f(x[i]),ls='--',color='r')
    ax.scatter(x[i],f(x[i]), marker='o',color='r', label=r'$N=%.0f$'%(i),s=50)                 
    ax1.scatter(i,Cost[i],marker='.',color='k')                  
    
    ax.legend()


Animation = animation.FuncAnimation(fig, Update, frames=N,init_func=init)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800 )
filename='/Users/juanpablomendozaarias/Library/Mobile Documents/com~apple~CloudDocs/Universidad/Tercer Semestre/Métodos Computacionales/MetodosI_JuanPabloMendoza/Parcial 3/Aprendizaje Automático/Gradient1D.mp4'
Animation.save( filename, writer=writer )

print(f'No es posible hallar el mínimo con precisión porque la función de costo cerca de este alterna entre un número positivo y un negativo (+-Learning R.)')
print(f'Cerca del mínimo el algoritmo empezará a oscilar entre dos valores')
print(f'Esto se debe a que la derivada alterna entre -1 y 1 alrededor del mínimo.')
print(f'Sin embargo, con un learning rate muy bajo puede minimizarse el rango de esta oscilación, situando en promedio al punto calculado muy cerca del mínimo, como se observa en el video.')
print(f'Con este algoritmo, usando un learning rate de 0.001, el mínimo se encuentra entre 1.99130614 y 2.00130614')
print(f'La desventaja es que si el punto seed no está muy cerca del mínimo, tardará mucho en llegar a el.')
print(f'Con learning rates más altos, se llegara a los alrededores del mínimo más rápido, pero la oscilación sobre este tendrá un mayor rango, haciéndolo más impreciso.')