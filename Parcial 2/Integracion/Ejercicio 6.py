import numpy as np
import sympy as sym

def Integral_Trapecio(funcion, a, b, N):
    h=(b-a)/N
    Intervalo = np.linspace(a,b,N)
    x_1 = Intervalo[0]
    Integral = 0.
    for i in range(1,Intervalo.size):
        x_2 = Intervalo[i]
        Integral += 0.5*h*(funcion(x_1)+funcion(x_2))
        x_1 = x_2
    return Integral

def Integral_Simpson(funcion, a, b, N):
    Intervalo = np.linspace(a,b,N)
    x_1 = Intervalo[0]
    Integral = 0
    for i in range(1,Intervalo.size):
        x_2 = Intervalo[i]
        x_m = 0.5*(x_1+x_2)
        h = x_m-x_1
        Integral += (1/3)*h*(funcion(x_1)+4*funcion(x_m)+funcion(x_2))
        x_1 = x_2
    return Integral
        

def funcion(x, R=0.5, a=0.01):
    resultado = np.sqrt((a**2)-(x**2))/(R+x)
    return resultado

def Integral_exacta_funcion(R=0.5, a=0.01):
    return np.pi*(R-np.sqrt((R**2)-(a**2)))

Integral_trapecio = Integral_Trapecio(funcion, -0.01,0.01, 1000000)
Integral_simpson = Integral_Simpson(funcion, -0.01, 0.01, 1000000)
print(Integral_simpson)
Integral_exacta = Integral_exacta_funcion()
Error_trapecio = 100*(1-Integral_trapecio/Integral_exacta)
Error_simpson = 100*(1-Integral_simpson/Integral_exacta)
print(f'Error Trapecio: {Error_trapecio}, \nError Simpson: {Error_simpson}')

