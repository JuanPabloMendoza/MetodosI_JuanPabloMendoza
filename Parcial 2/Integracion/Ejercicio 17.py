import numpy as np


def Integral_GL_corta(funcion, n):
    # Más rápida. Usa los pesos y raices dados por numpy
    def funcion_corregida(x):
        return funcion(x) * np.exp(x)
    raices,pesos = np.polynomial.laguerre.laggauss(n)
    suma = 0
    for i in range(n):
        suma += pesos[i] * funcion_corregida(raices[i])
    return suma

def f(x):
    return (x**3)/(np.exp(x)-1)



Int = Integral_GL_corta(f, 3)
print(Int)
Real = (np.pi**4)/15
print(Real)
print(f'Accuracy = {Int/Real}')

