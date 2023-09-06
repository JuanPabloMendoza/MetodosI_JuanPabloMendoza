import numpy as np


def derivada_central(funcion, x, h=0.001):
    return (funcion(x+h)-funcion(x-h))/(2*h)


def GetNewtonRaphson(x, funcion, derivada = derivada_central, precision=1*10**(-8), itmax=1000):
    
    xn=x
    error = 1.
    it=0
    
    while (error > precision) and (it < itmax):
        df_xn = derivada(funcion, xn)
        if df_xn != 0:
            xn_ = xn - (funcion(xn) / df_xn )
            xn=xn_
        
        error = np.abs( funcion(xn) / derivada(funcion,xn) )
        it+=1
    
    return xn


def polinomio(x):
    return 3*x**5 + 5*x**4 - x**3


Intervalo_tentativo = [-5,5]
h = 0.05
N = int((Intervalo_tentativo[1] - Intervalo_tentativo[0]) / h)
X = np.linspace(Intervalo_tentativo[0], Intervalo_tentativo[1], N)

def raices_X(X, funcion):
    Y = np.array([0]*X.size, dtype='float64')
    for i in range(X.size):
        Y[i] = GetNewtonRaphson(X[i], funcion)
    return Y

def descartar_raices(raices, h_, dec):
    raices_definitivas = []
    raices.round(4)
    xi = raices[0]
    for i in range(1,np.size(raices)):
        xf = raices[i]
        if np.abs(xf-xi)>h_:
            if round(xi,dec) not in raices_definitivas:
                raices_definitivas.append(round(float(xi),dec))
        xi=xf
    raices_definitivas.sort()
    return raices_definitivas

def raices_polinomio(polinomio, X, h_, dec):
    raices = raices_X(X, polinomio)
    print(descartar_raices(raices, h_, dec))
    
raices_polinomio(polinomio, X, 0.1, 6)           
        