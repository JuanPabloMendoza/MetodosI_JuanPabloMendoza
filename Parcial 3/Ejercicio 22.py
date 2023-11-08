import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import time
init_printing(use_unicode=True)

#-------
m = 11.46 #kg
g = 10 #m/s^2
q = 3e-4 #C
e_0 = 8.8541878176e-12 #C^2 /(N·m²).
L = 5 #m
alpha = ((0.25+(np.sqrt(2)/2))*(1/(m*g))*((q**2)/(4*np.pi*e_0*(L**2))))
#----
x = Symbol('x',real=True)
y = Symbol('y',real=True)
z = x+I*y


f = lambda z, k=alpha: (sin(z))**6 +(k*sin(z))**2 - k**2

F = [re(f(z)), im(f(z))]

Jacobiano = Matrix([[diff(F[0],x,1),diff(F[0],y,1)], [diff(F[1],x,1),diff(F[1],y,1)]])
InversaJacobiano = Jacobiano.inv()

IJ = lambdify([x,y], InversaJacobiano, 'numpy')
F = lambdify([x,y], F, 'numpy')

def NewtonRaphson(z,F,IJ,itmax=100,precision=1e-10):
    error = 1
    it = 0
    
    while error > precision and it < itmax:
        
        
        z1 = z - np.dot(IJ(z[0], z[1]),F(z[0], z[1]))
        
        diff = np.array([z1-z], dtype='float64')
        error = np.max( np.linalg.norm(diff))
        z = z1
        it +=1
    return np.round(z,7)

#Calculamos sobre un círculo (discreto) de radio 1 las soluciones a la ecuación
#Sabemos que están distribuidas sobre un círculo en el plano complejo.
N=1000
roots = np.zeros((N,2))
x_= np.linspace(0,2*np.pi,N)

def MultipleRoots1(x, newtonmethod, f, df, roots):
    for i in range(x.shape[0]):
        x_ = [np.cos(x[i]), np.sin(x[i])]
        root = newtonmethod(x_, f, df)
        roots[i,0] = root[0]
        roots[i,1] = root[1]

MultipleRoots1(x_, NewtonRaphson, F, IJ, roots)


#Tomamos modulo 2pi en las partes reales para eliminar soluciones equivalentes.
roots = np.unique(roots, axis=0)
roots = roots[roots[:, 1].argsort()]
for i in range(roots.shape[0]):
    roots[i,0] = np.round(roots[i,0]%(2*np.pi),5)

roots = np.unique(roots, axis=0)

#Se complexifican las soluciones y se extraen las reales.
R = np.empty((roots.shape[0]), dtype='complex64')
Rreal = np.empty((0),dtype='float64')
for i in range(roots.shape[0]):
    R[i] = roots[i,0]+1j*roots[i,1]
    if roots[i,1] == 0j:
        Rreal = np.append(Rreal,roots[i,0])

#-----
def impr(lista):
    for i in range(lista.shape[0]):
        print(lista[i])
    print('\n')
#-----

Rango_aceptado = (0., np.pi/2)

#Se extraen las soluciones en el rango aceptado
Rreales_aceptadas = np.empty((0),dtype='float64')
for sol in Rreal:
    if sol>Rango_aceptado[0] and sol<Rango_aceptado[1]:
        Rreales_aceptadas = np.append(Rreales_aceptadas, sol)


print(f'Las soluciones (reales y complejas) al problema son (en radianes y modulo 2pi):')
impr(R)

print(f'Las soluciones reales en grados son:')
impr(Rreal*180/np.pi)

print(f'La solución aceptada para el ejercicio (en grados) es:')
impr(Rreales_aceptadas*180/np.pi)

print(f'Claramente, al ser un polinomio en términos de la función seno, que tiene periodo 2pi, habrán infinitas soluciones. Estas se obtienen sumando 2pi*k (o 360*k en grados) con k entero a cada una de las soluciones mostradas.')