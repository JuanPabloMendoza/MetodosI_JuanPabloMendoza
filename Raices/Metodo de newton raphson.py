import numpy as np
import matplotlib.pyplot as plt

h=0.001
def Function(x):
    return 5*(1-np.exp(-x))-x

def Function2(x):
    return 0.5*((5*x**3)-3*x)
def derivada_central(funcion, x, h):
    derivada_x = ((funcion(x+h)-funcion(x-h))/(2*h))
    return derivada_x



def GetNewtonMethod(f,df,xn,itmax=100000,precision=1e-5):
    
    error = 1.
    it=0
    
    while error > precision and it<itmax:
        try:
            xn1 = xn - f(xn)/df(f,xn,h)
            error = np.abs(f(xn)/df(f,xn,h))
        except ZeroDivisionError:
            print('Division por cero')
        xn=xn1
        it+=1
    return xn
        
N=20
root = GetNewtonMethod(Function, derivada_central, 0)
roots = np.empty(N)
x= np.linspace(-1,1, 20)

def roots_each_x(x, newtonmethod, f, df, roots):
    for i in range(N):
        x_ = x[i]
        root = newtonmethod(f, df, x_)
        roots[i] = root 
        
roots_each_x(x, GetNewtonMethod, Function2, derivada_central, roots)
print(roots)

real_roots = []
root_i = roots[0]
e=0.001
for i in range(roots.size):
    difference = root_i-roots[i]
    if not np.abs(difference)<e:
        real_roots.append(root_i)
    root_i= roots[i]
print(real_roots)
    
        
            