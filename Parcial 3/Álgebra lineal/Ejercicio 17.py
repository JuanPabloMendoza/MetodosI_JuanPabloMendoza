import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import time
init_printing(use_unicode=True)

x = Symbol('x',real=True)
y = Symbol('y',real=True)
z = x+I*y

f = lambda z: (z)**3 -1
F = [re(f(z)), im(f(z))]

Jacobiano = Matrix([[diff(F[0],x,1),diff(F[0],y,1)], [diff(F[1],x,1),diff(F[1],y,1)]])
InversaJacobiano = Jacobiano.inv()
IJ = lambdify([x,y], InversaJacobiano, 'numpy')
F = lambdify([x,y], F, 'numpy')


def NewtonRaphson(z,F,IJ,itmax=50,precision=1e-7):
    error = 1
    it = 0
    
    while error > precision and it < itmax:
        
        
        z1 = z - np.dot(IJ(z[0], z[1]),F(z[0], z[1]))
        
        diff = np.array([z1-z], dtype='float64')
        error = np.max( np.linalg.norm(diff))
        z = z1
        it +=1
    return z

raices = [np.array([-0.5, 0.866],dtype='float64'),
              np.array([-0.5, -0.866],dtype='float64'),
              np.array([1, 0.],dtype='float64')]

def color(seed, F, IJ, raices,err=1e-3):
    raiz = NewtonRaphson(seed,F, IJ)
    diff = raiz - raices
    if np.linalg.norm(diff[0]) < err:
        return 20
    elif np.linalg.norm(diff[1]) < err:
        return 100
    else:
        return 255

N=300
X = np.linspace(-1,1,N)
Y = np.linspace(-1,1,N)
Fractal = np.zeros((N,N), np.int64)

for i in range(N):
    for j in range(N):
        Fractal[i,j] = color([X[i],Y[j]], F, IJ,raices)
plt.imshow(Fractal, cmap='coolwarm' ,extent=[-1,1,-1,1])
plt.show()


   
