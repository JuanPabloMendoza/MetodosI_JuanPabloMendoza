import numpy as np
import matplotlib.pyplot as plt



Intervalo = [0.1, 1.1]
h = 0.01
N = int((Intervalo[1] - Intervalo[0]) / h)

#FUNCIONES
def sqrt_tanx(x):
    return np.sqrt(np.tan(x))

def derivada_sqrt_tanx(x):
    return ((1/np.cos(x))**2)/(2*(np.sqrt(np.tan(x))))

def print_funcion(X,Y):
    for i in range(np.size(X)):
        print([X[i], Y[i]])


#DATOS 

X = np.linspace(Intervalo[0], Intervalo[1], N)
Y = sqrt_tanx(X)

#DERIVADAS 

def derivada_progresiva(funcion, x, h):
    return (1/(2*h))*((-1)*3*funcion(x)+4*funcion(x+h)-funcion(x+2*h))

def derivada_progresiva_ptos_arbitrarios(p0, p1, p2, h):
    # ¿cómo se define h si los puntos no son equidistantes?
    return (1/(2*h))*((-1)*3*p0[1]+4*p1[1]-p2[1])

def derivada_central(funcion, x, h):
    return (funcion(x+h)-funcion(x-h))/(2*h)


#DERIVADA PROGRESIVA Y REAL DE LA FUNCION 

Y_df_progresivo = derivada_progresiva(sqrt_tanx, X, h)
Y_df_central = derivada_central(sqrt_tanx, X, h)
Y_df_real = derivada_sqrt_tanx(X)

print('Derivada progresiva')
print_funcion(X,Y_df_progresivo)
print('Derivada central')
print_funcion(X,Y_df_central)

#ERROR
Error_derivada_progresiva = np.abs((Y_df_progresivo - Y_df_real) * 100 /  Y_df_real)
Error_derivada_central = np.abs(( Y_df_central - Y_df_real)  * 100 /  Y_df_real)

#GRÁFICAS

fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_title('Derivadas')
ax1.scatter(X, Y_df_progresivo, label = 'Derivada progresiva')
ax1.scatter(X, Y_df_central, label = 'Derivada central', color = 'red', marker='.')
ax1.scatter(X, Y_df_real, label = 'Derivada real', color = 'grey', marker='.')
ax1.legend()

ax2.set_title('Error')
ax2.scatter(X, Error_derivada_progresiva, label = 'Derivada progresiva')
ax2.scatter(X, Error_derivada_central, label = 'Derivada central', color = 'red', marker='.')
ax2.legend()

plt.show()

#COMENTARIOS

comentario_1 = 'El error de ambas derivadas se asemeja el valor del error predecido, O(h^2), que en este caso es aproximadamente 0.0001, tan solo 10 veces menos de lo observado.'
comentario_2 = 'Ambas mantienen el mismo orden de precision.'
print(comentario_1, comentario_2)









