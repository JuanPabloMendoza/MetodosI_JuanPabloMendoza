import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym


# INTERPOLACION DE LAGRANGE
def Polinomio_lagrange(x, puntos_x, i):
    n = np.size(puntos_x)
    polinomio = 1
    for j in range(n):
        if j!= i:
            polinomio *= (x - puntos_x[j]) / (puntos_x[i] - puntos_x[j])
    return polinomio

def Interpolador(X, puntos_x, puntos_y):
    n = np.size(puntos_x)
    suma = 0
    for i in range(n):
        suma += puntos_y[i]*Polinomio_lagrange(X, puntos_x, i)
    return suma

#DATA

def extraer_datos(link):
    df = pd.read_csv(link, sep=',')
    print(df.values)
    return df

datos = extraer_datos('https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/Parabolico.csv')
puntos_x = datos['X'].values
puntos_y = datos['Y'].values


# TRAYECTORIA

Intervalo = [0,6.7]
h = 0.0001
N = int((Intervalo[1] - Intervalo[0]) / h)
X = np.linspace(Intervalo[0], Intervalo[1], N+1)

Polinomio = Interpolador(X, puntos_x, puntos_y)


def getRaices_Polinomio_2_grado(Polinomio, X, e=0.001):
    raiz = float('inf')
    i_y_maximo = 0
    for i in range(X.size):
        if Polinomio[i]>Polinomio[i_y_maximo]:
            i_y_maximo = i
    
    for i in range(i_y_maximo, int(X.size)):
        if np.abs(Polinomio[i])<e:
            raiz = X[i]
            e=np.abs(Polinomio[i])
            print(Polinomio[i])
    if raiz==0:
        return None
    print(f'Error: {Polinomio[i]}')
    return raiz


def Angulo_velocidad_inicial(Polinomio, X, g=9.8):
    angulo = np.degrees( np.arctan((Polinomio[2]-Polinomio[0])/(X[2]-X[0])) )
    Distancia_maxima = getRaices_Polinomio_2_grado(Polinomio, X)
    velocidad_inicial = np.sqrt( (g * Distancia_maxima) / np.sin(2*np.radians(angulo)))
    return angulo, velocidad_inicial


angulo, velocidad = Angulo_velocidad_inicial(Polinomio, X)
print(f'Angulo: {angulo}, Velocidad: {velocidad}')

#GRAFICAS

plt.scatter(puntos_x, puntos_y)
plt.scatter(X, Polinomio, marker = '.', label = f'Ang: {angulo}, Vo: {velocidad}')
plt.legend()
plt.show()