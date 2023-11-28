import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

def f(r):
    x=r[0]
    y=r[1]
    z=r[2]
    return x**2 + y**2 + z**2 -2*z + 1

r0 = [1.,1.,1.]
constraints = ( {'type':'eq','fun': lambda p: 2*p[0] + -4*p[1] + 5*p[2] - 2} )
result = spo.minimize( f, x0=r0, constraints=constraints)
print(f'El mínimo de la función se encuentra en {result.x} y su valor es {np.round(result.fun,3)}')