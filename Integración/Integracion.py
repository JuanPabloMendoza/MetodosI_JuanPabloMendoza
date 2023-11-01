import numpy as np

def IntegralTrapecio(funcion, a, b, N):
    h=(b-a)/N
    Intervalo = np.linspace(a,b,N)
    x_1 = Intervalo[0]
    Integral = 0.
    for i in range(1,Intervalo.size):
        x_2 = Intervalo[i]
        Integral += 0.5*h*(funcion(x_1)+funcion(x_2))
        x_1 = x_2
    return Integral

def IntegralSimpson(funcion, a, b, N):
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

def IntegralLegendre(funcion,a,b,n):
    raices, pesos = np.polynomial.legendre.leggauss(n)
    raices = 0.5*(raices*(b-a)+(b+a))
    Integral = 0
    for i in range(n):
        #raiz_con_sustitucion = 0.5*(raices[i]*(b-a)+(b+a))
        Integral += pesos[i]*funcion(raices[i])
    return 0.5*(b-a)*Integral
    

def IntegralLaguerre(funcion, n):
    # Más rápida. Usa los pesos y raices dados por numpy
    def funcion_corregida(x):
        return funcion(x) * np.exp(x)
    Roots,Weights = np.polynomial.laguerre.laggauss(n)
    suma = 0
    for i in range(n):
        suma += Weights[i] * funcion_corregida(Roots[i])
    return suma

def IntegralHermite(funcion, n):
    # Más rápida. Usa los pesos y raices dados por numpy
    def funcion_corregida(x):
        return funcion(x) * np.exp(x**2)
    raices,pesos = np.polynomial.hermite.hermgauss(n)
    suma = 0
    for i in range(n):
        suma += pesos[i] * funcion_corregida(raices[i])
    return suma

def Doble_Integral_Trapecio(funcion, a, b, c, d, N, R):
    X = np.linspace(a,b,N+1)
    Y = np.linspace(c,d,N+1)
    
    def Promedio(funcion, Cuadrado, R):
        
        Suma = 0
        for Vertice in Cuadrado:
            #Generalizacion para cualquier funcion
            if not np.isnan(funcion(Vertice[0], Vertice[1])):
                Suma += funcion(Vertice[0], Vertice[1])
            """ if (Vertice[0]**2) + (Vertice[1]**2) < R**2:
                Suma += funcion(Vertice[0], Vertice[1], R) """
        return Suma/4
    
    Area_cuadrado = (X[1]-X[0])*(Y[1]-Y[0])
    
    Resultado = 0
    for i in range(X.size-1):
        for j in range(Y.size-1):
            Cuadrado = np.array([(X[i], Y[j]), (X[i+1], Y[j]), (X[i], Y[j+1]), (X[i+1], Y[j+1])])
            Resultado += Area_cuadrado*Promedio(funcion, Cuadrado, R)
    return Resultado

def Doble_Integral_Legendre(funcion,a,b,c,d,n):
    raices, pesos = np.polynomial.legendre.leggauss(n)
    raices_x = 0.5*(raices*(b-a)+(b+a))
    raices_y = 0.5*(raices*(d-c)+(d+c))
    Doble_Integral = 0
    for i in range(n):
        Integral_int = 0
        for j in range(n):
            Integral_int += pesos[j]*funcion(raices_x[i], raices_y[j])
        Doble_Integral += pesos[i]*Integral_int
    return 0.25*(b-a)*(d-c)*Doble_Integral

def Doble_Integral_Laguerre(funcion, n):
    def funcion_corregida(x,y):
        return funcion(x,y) * np.exp(x+y)
    Roots, Weights = np.polynomial.laguerre.laggauss(n)
    Doble_Integral = 0
    for i in range(n):
        Integral_int = 0
        for j in range(n):
            Integral_int += Weights[j]*funcion_corregida(Roots[i], Roots[j])
        Doble_Integral += Weights[i]*Integral_int
    return Doble_Integral

def Doble_Integral_Hermite(funcion, n):
    def funcion_corregida(x,y):
        return funcion(x,y) * np.exp((x**2)+(y**2))
    Roots, Weights = np.polynomial.hermite.hermgauss(n)
    Doble_Integral = 0
    for i in range(n):
        Integral_int = 0
        for j in range(n):
            Integral_int += Weights[j]*funcion_corregida(Roots[i], Roots[j])
        Doble_Integral += Weights[i]*Integral_int
    return Doble_Integral


