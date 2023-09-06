import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

def funcion(x):
    return np.exp(-x)-x



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

def coeficientes_parabola(coeficientes, X):
    x_0 = X[0]
    x_1 = X[1]
    x_2 = X[2]
    a = coeficientes[2]
    b = coeficientes[1] - (x_0+x_1) * a
    c = coeficientes[0] - x_0 * coeficientes[1] + x_0 * x_1 * coeficientes[2]
    return a,b,c


#d
def x_3_muller(a,b,c):
    if b<0:
        return -2*c/(b-np.sqrt((b**2)-4*a*c))
    else:
        return -2*c/(b+np.sqrt((b**2)-4*a*c))


#a
def algoritmo_muller(X_2, Y_2, funcion, precision, N):
    #X_2, Y_2 contienen los dos puntos iniciales
    #X_3 tiene un nuevo punto
    x_0 = X_2[0]
    y_0 = Y_2[0]
    x_1 = X_2[1]
    y_1 = Y_2[1]
    
    #c
    x_2 = (x_0+x_1)/2
    y_2 = funcion(x_2)

    error=1.
    it=1
    while error > precision and it<N:
        X_3 = np.array([x_0, x_1, x_2])
        Y_3 = np.array([funcion(X_3[0]), funcion(X_3[1]), funcion(X_3[2])])
        a,b,c = coeficientes_parabola(coeficientes(X_3, Y_3), X_3)
        x_3 = x_3_muller(a, b, c)
        error = np.abs(funcion(X_3[2]))
        x_0 = x_1
        x_1 = x_2
        x_2 = x_3
        it+=1

    return x_3, error, it
    
    


#b. Viendo la grafica de la funcion, se escogen los puntos X[0] y X[1] porque funcion(X[0])*funcion(X[1])<0
X_2 = np.array([0,2])
Y_2 = np.array([funcion(X_2[0]), funcion(X_2[1])])

#e
criterio_parada = 1*10**(-10)
N = 100

#f
raiz, error, it = algoritmo_muller(X_2, Y_2, funcion, criterio_parada, N)
print(f'La raiz es X={raiz}. Error: {error}. Iteraciones: {it}')


#GRAFICA FUNCION Y SU CERO
X = np.linspace(-4, 4, 100)
Y = funcion(X)
zeros = X*0

plt.plot(X, Y)
plt.scatter([raiz], [funcion(raiz)], label = f'Cero={raiz}', marker = '.')
plt.plot(X, zeros, color = 'black')
plt.plot([0]*200, np.linspace(funcion(X[0]), funcion(X[np.size(X)-1]), 200), color = 'black')
plt.legend()
plt.show()



        