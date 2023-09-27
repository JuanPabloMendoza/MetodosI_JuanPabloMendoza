import numpy as np
import matplotlib.pyplot as plt

def Seno(x):
    return np.sin(x)

def Cos(x):
    return np.cos(x)

def ExactDerivative(x):
    return np.cos(x)

N=20
x=np.linspace(0,2*np.pi, N)
print(x)
valor_funcion_seno=Seno(x)
valor_funcion_cos=Cos(x)
""" plt.scatter(x,valor_funcion)
plt.show() """


h=0.00001

def derivada_derecha(funcion, x,h):
    derivada_x = ((funcion(x+h)-funcion(x))/h)
    return derivada_x

def derivada_izquierda(funcion, x,h):
    derivada_x = ((funcion(x)-funcion(x-h))/h)
    return derivada_x

def derivada_central(funcion, x, h):
    derivada_x = ((funcion(x+h)-funcion(x-h))/(2*h))
    return derivada_x

""" valor_derivada_derecha = derivada_derecha(Seno, x, h)
valor_derivada_izquierda = derivada_izquierda(Seno, x, h)
valor_derivada_central = derivada_central(Seno, x, h)


ErrorR = np.abs(valor_derivada_derecha-valor_funcion_cos)
ErrorI = np.abs(valor_derivada_izquierda-valor_funcion_cos)
ErrorC = np.abs(valor_derivada_central-valor_funcion_cos)
plt.scatter(x,ErrorR, label='Error Derecho')
plt.scatter(x,ErrorI, label='Error Izquierdo')
plt.scatter(x,ErrorC, label='Error Central')
plt.legend()
plt.show()

plt.scatter(x,valor_derivada_derecha, label='Derivada Derecha')
plt.scatter(x,valor_derivada_izquierda, label='Derivada Izquierda')
plt.scatter(x,valor_derivada_central, label='Derivada Central')
plt.scatter(x,valor_funcion_seno)
plt.scatter(x,valor_funcion_cos)
plt.legend()
plt.show() """


def recta(x):
    return 3*x+5

x2 = np.linspace(-10,3,1000000)

def distancia(x):
    return np.sqrt((x**2)+(3*x+5)**2)

y2 = distancia(x2)
plt.scatter(x2,y2)
plt.show()

derivada = derivada_central(distancia, x2,h)
plt.scatter(x2,derivada)
plt.show()

def cero(y,x):
    signo_i = y[0]/np.abs(y[0])
    for i in range(y.size):
        signo_f = y[i]/np.abs(y[i])
        if signo_f != signo_i:
            return (x[i],y[i])

print(cero(derivada,x2))

