import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
h=0.0001
L = (lambda x: 0.5*x*(x-1), lambda x: -(x-1)*x, lambda x: 0.5*x(x+1))
print(L[0](-1))
print([])

def multiplicador(funcion_1, funcion_2):
    def funcion(x,y):
        return funcion_1(x,y)*funcion_2(x,y)
    return funcion
    
    
""" def polinomio(lista_valores):
    n=len(lista_valores)
    polinomios_base = [1]*n
    for i in range(n):
        lista_i = []
        for j in range(n):
            if i != j:
                lista_i.append(lista_valores[j])
        lista_monomios = []
        for j in range(n-1):
            lista_monomios.append(lambda x: (x-lista_i[j]))
        for j in range(n-1):
            def polinomio_base_j(x,y):
                lista_monomios(x,y)
            polinomios_base[i]= lambda x: (polinomios_base[i])*(lista_monomios[j])
    return polinomios_base

polinomio_1 = polinomio((-1,0,0))
print(polinomio_1[0](-1)) """


def Lagrange(x,X,i):
    L = 1.
    for j in range(X.shape[0]):
        if i != j:
            L*=(x-X[j])/(X[i]-X[j])
    return L

def Interpolate(x,X,Y):
    
    Poly = 0.
    for i in range(X.shape[0]):
        Poly += Lagrange(x,X,i)*Y[i]
    return Poly
    
X=np.array([-2,-1,0,1])
Y=np.array([0.5,10,5,8])
x0=np.linspace(-2,1,10)
y0=Interpolate(x0,X,Y)
plt.scatter(X,Y)
plt.scatter(x0,y0)
#plt.show()


x=sym.Symbol('x', real=True)
f=Interpolate(x,X,Y)
f=sym.simplify(f)
print(f)


    

def funcion(x):
    return (3.75*x**3)+(4*x**2)-(4.75*x)+5

def index_xn_cercano(xn,X):
    index=0
    for i in range(X.shape[0]):
        df = xn-x[i]
        if df<0:
            index=i
            break
    return index

def f_xn_cercano(xn,X,Y):
    index = index_xn_cercano(xn,X)
    return Y[index]

def derivada_central_discreta(x,X,Y,h):
    index_x_X=index_xn_cercano(x,X)
    xf=X[index_x_X+1]
    xi=X[index_x_X-1]
    h=xf-xi
    derivada_x = (Y[index_x_X+1]-Y[index_x_X-1])/(2*h)
    return derivada_x

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

def GetNewtonMethod_discreto(X,Y,df,xn,itmax=100,precision=1e-5):
    error = 1.
    it=0
    suma=0
    while error > precision and it<itmax:
        try:
            i=index_xn_cercano(xn,X)
            xn1 = xn - Y[i]/df(xn,X,Y,h)
            error = np.abs(Y[i]/df(xn,X,Y,h))
        except ZeroDivisionError:
            print('Division por cero')
        xn=xn1
        it+=1
        suma+=xn
        print(it)
    return suma/10000000
        
x=np.linspace(-20,20,200000)
derivada_aprox = derivada_central(funcion,x,h)
print(derivada_aprox)
root = GetNewtonMethod_discreto(x,derivada_aprox, derivada_central_discreta,0.5)
print(root)


