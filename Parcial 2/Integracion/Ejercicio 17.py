import numpy as np
import matplotlib.pyplot as plt 

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



Real = (np.pi**4)/15

#a
n=3
Int = Integral_GL_corta(f, 3)
print(Int)

#b
N=np.arange(2,11)
Accuracy = np.zeros(9)
print(N)
for i in range(9):
    Int = Integral_GL_corta(f, N[i])
    Accuracy[i] = Int/Real

plt.scatter(N, Accuracy, color = 'blue', label = 'Laguerre Quadrature Accuracy')
plt.xlabel('n')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.show()


