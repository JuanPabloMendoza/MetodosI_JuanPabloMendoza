import numpy as np


def producto_punto(a,b):
    dot = 0
    for i in range(2):
        dot += a[i]*b[i]
    return dot
print(producto_punto([1,2], [3,4]))

def producto_cruz(a,b):
    if len(a) == 2:
        a.append(0)
    if len(b) == 2:
        b.append(0)
    return [a[1]*b[2]-a[2]*b[1], a[0]*b[2]-a[2]*b[0], a[0]*b[1]-a[1]*b[0]]
 
print(producto_cruz([1,0,0], [0,1,0]))

            