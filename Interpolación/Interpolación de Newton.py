import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

#FUNCIONES

def coeficientes(X,Y):
    """
    Genera una matriz de coeficientes cuyo elemento ij es la diferencia dividida ij
    que se calcula como Fij = (F(i-1)(j+1) - F(i-1)j) / (X[i+j] - X[j])
    Los elementos de la primera columna seran los coeficientes a multiplicar por los polinomios
    calculados en la interpolacion. Estos se guardan en coef.
    """
    n_coeficientes = np.shape(X)[0]
    
    coef = np.zeros(n_coeficientes)
    matriz_f = np.zeros((n_coeficientes,n_coeficientes))
    
    matriz_f[0]=Y
    coef[0] = Y[0]
    
    for i in range(1,n_coeficientes):
        for j in range(n_coeficientes-i):
            matriz_f[i][j] = (matriz_f[i-1][j+1]-matriz_f[i-1][j])/(X[i+j]-X[j])
        coef[i] = matriz_f[i][0]
    return coef




def multiplicador(x, X, i):
    """
    Realiza la multiplicacion de los monomios.
    Dado el parametro i, multiplicador realiza la siguiente operacion
    (x - X[0]) * ... * (x-X[i-1])
    """
    x_nuevo = 1
    for j in range(i):
        x_nuevo *= (x-X[j])
    return x_nuevo


### PRUEBAS

intervalo = [0,5]
n=100

valores_x = np.linspace(intervalo[0], intervalo[1], n)
X = np.array((1,2,3,4), dtype='float64')
Y = np.array((6,9,2,5), dtype='float64')
print(X,Y)

coeficientes(X,Y)
poly = Interpolador(valores_x, X,Y)

#GRAFICA
plt.scatter(valores_x,poly, color = 'blue')
plt.scatter(X,Y, color='red')
plt.show()


#EXPRESION SIMBOLICA
x = sym.Symbol('x',real=True)
y = Interpolador(x,X,Y)
y = sym.simplify(y)
print(y)

            
        
        