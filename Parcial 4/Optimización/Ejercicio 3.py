import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

def V(r):
    x = r[0]
    y = r[1]
    z = r[2]
    return -x*y*z

r0 = [1., 1., 1.]
constraints = ( {'type':'eq','fun': lambda p: p[0]*p[1]+2*p[1]*p[2]+2*p[0]*p[2] - 12} )
result = spo.minimize( V, x0=r0, constraints=constraints)
print(f'd: El volumen máximo sujeto a la restricción será de {np.round(-result.fun)} cm3')


