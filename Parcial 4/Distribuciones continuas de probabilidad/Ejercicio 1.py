import numpy as np
import scipy.optimize as spo
import scipy.integrate as integrate
import sympy as sym

x=sym.Symbol('x', real=True)
y=sym.Symbol('y', real=True)

def f(x,y):
    if (0<=x and x<=1) and (0<=y and y<=1):
        return (2/3)*(x+2*y)
    else:
        return 0




#a
def f_(r):
    #se reescribe la funcion para poder usar optimize
    x=r[0]
    y=r[1]
    if (0<=x and x<=1) and (0<=y and y<=1):
        return (2/3)*(x+2*y)
    else:
        return 0
bounds = ((0,1),(0,1))
min = spo.minimize(f_, x0=[0.,0.],bounds=bounds).fun
print(f'Fuera del cuadrado unitario, la función es 0 (no negativo), y dentro el mínimo de es {min}. Por tanto, la función es no negativa en ningún punto.')

I = integrate.dblquad(f, -np.inf, np.inf, -np.inf, np.inf)[0]
print(f'La integral desde x=-inf hasta x=inf y desde y=-inf hasta y=inf es {np.round(I,3)}')

