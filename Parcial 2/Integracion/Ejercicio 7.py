import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

def Integral_Doble(funcion, a, b, c, d, N, R):
    X = np.linspace(a,b,N+1)
    Y = np.linspace(c,d,N+1)
    
    def Promedio_condicion_esfera(funcion, Cuadrado, R):
        #Revisar cómo generalizar para cualquier funcion
        Suma = 0
        for Vertice in Cuadrado:
            if (Vertice[0]**2) + (Vertice[1]**2) < R**2:
                Suma += funcion(Vertice[0], Vertice[1])
        return Suma/4
    
    Area_cuadrado = (X[1]-X[0])*(Y[1]-Y[0])
    
    Resultado = 0
    for i in range(X.size-1):
        for j in range(Y.size-1):
            Cuadrado = np.array([(X[i], Y[j]), (X[i+1], Y[j]), (X[i], Y[j+1]), (X[i+1], Y[j+1])])
            Resultado += Area_cuadrado*Promedio_condicion_esfera(funcion, Cuadrado, R)
    return Resultado

def Esfera(x,y, R=1):
    return np.sqrt((R**2)-(x**2)-(y**2))

R=1
print(Integral_Doble(Esfera, -1, 1, -1, 1, 1000, R))
print((2/3)*np.pi*R**3)

#plot esfera
""" def esfera(X,Y,Z):
    Puntos = np.empty((0,3))
    for x in X:
        for y in Y:
            for z in Z:
                if (x**2)+(y**2)+(z**2)<=1 and (x**2)+(y**2)+(z**2)>=0.999 :
                    coord = np.array([x,y,z])
                    Puntos = np.r_[Puntos, [coord]]
    return Puntos

Puntos = esfera(X,Y,Z)
fig = plt.figure()
ax2 = fig.add_subplot(projection='3d')
ax2.scatter(Puntos[:,0], Puntos[:,1], Puntos[:,2], color='blue')
plt.show() """
