import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)
#Cantidades estandar
N_a = 6.0225*(10**23)
k = 1.3805*(10**(-23))
masa_helio = 0.004
masa_nitrogeno = 0.028
masa_xenon_132 = 0.1319041535

#GAUSS-LAGUERRE
def GetLaguerreRecursive(n,x):
    if n==0:
        poly = sym.Pow(1,1)
    elif n==1:
        poly = 1-x
    else:
        poly = ((2*n-1-x)*GetLaguerreRecursive(n-1,x)-(n-1)*GetLaguerreRecursive(n-2,x))/n
   
    return sym.simplify(poly)
n=4
print(f'Ejercicio 3.1.1\n Polinomio de laguerre de grado {n}: {GetLaguerreRecursive(n,x)}\n')

#RAICES

def GetDLaguerre(n,x):
    Pn = GetLaguerreRecursive(n,x)
    return sym.diff(Pn,x,1)

def GetNewton(f,df,xn,itmax=10000,precision=1e-10):
    ##revisar precision ######
    
    error = 1.
    it = 0
    
    while error >= precision and it < itmax:
        
        try:
            
            xn1 = xn - f(xn)/df(xn)
            
            error = np.abs(f(xn)/df(xn))
            
        except ZeroDivisionError:
            print('Zero Division')
            
        xn = xn1
        it += 1
        
    if it == itmax:
        return False
    else:
        return xn
    
def GetRootsGLag(f,df,x,tolerancia = 11):
    
    Roots = np.array([])
    
    for i in x:
        
        root = GetNewton(f,df,i)
        croot = np.round( root, tolerancia )
        
        if croot not in Roots:
            Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots

def GetAllRootsGLag(n):
    a = 0
    b = n+(n-1)*np.sqrt(n)
    xn = np.linspace(a,b,100)
    
    Laguerre = []
    DLaguerre = []
    
    for i in range(n+1):
        Laguerre.append(GetLaguerreRecursive(i,x))
        DLaguerre.append(GetDLaguerre(i,x))
    
    poly = sym.lambdify([x],Laguerre[n],'numpy')
    Dpoly = sym.lambdify([x],DLaguerre[n],'numpy')
    Roots = GetRootsGLag(poly,Dpoly,xn)
    
    return Roots
n=4
print(f'Ejercicio 3.1.2\n Raices del polinomio de Laguerre de grado {n}: {GetAllRootsGLag(n)}\n')

def GetWeightsGLag(n):
    Roots = GetAllRootsGLag(n)

    

    Laguerre = []
    
    for i in range(n+2):
        Laguerre.append(GetLaguerreRecursive(i,x))
    
    poly = sym.lambdify([x],Laguerre[n+1],'numpy')
    Weights = []
    for root in Roots:
        Weights.append(root/(((n+1)**2)*(poly(root))**2))
    
    return Weights

n=4
print(f'Ejercicio 3.1.3\n Pesos del polinomio de Laguerre de grado {n}: {GetWeightsGLag(n)}\n')


#INTEGRAL 0 A INF

def Integral_GL(funcion, n):
    def funcion_corregida(x):
        return funcion(x) * np.exp(x)
    raices = GetAllRootsGLag(n)
    pesos = GetWeightsGLag(n)
    suma = 0
    for i in range(n):
        suma += pesos[i] * funcion_corregida(raices[i])
    return suma


def Integral_GL_corta(funcion, n):
    # Más rápida. Usa los pesos y raices dados por numpy
    def funcion_corregida(x):
        return funcion(x) * np.exp(x)
    raices,pesos = np.polynomial.laguerre.laggauss(n)
    suma = 0
    for i in range(n):
        suma += pesos[i] * funcion_corregida(raices[i])
    return suma

#EJERCICIOS 3.3.

#1
def P(v, T=300, m=0.001, R=k):
    M = m/N_a
    return 4 * np.pi * (( M / ( 2 * np.pi * R * T ))**(3/2)) * (v**2) * np.exp(-(M * (v**2))/( 2 * R * T))

def P_modificada(x):
    return (2/np.sqrt(np.pi))*np.sqrt(x)*np.exp(-x)

print(f'Ejercicio 3.3.1\nEl valor de la integral de la funcion de distribucion de velocidades entre 0 e infinito, realizando la sustitucion u=mv^2/2kT es {Integral_GL_corta(P_modificada, 15)}\n')

#2

v = np.linspace(0, 10000, 10000)
temperaturas = np.array([50, 100, 200, 300, 500, 600, 1000, 2500, 5000, 10000], dtype='float64')
Masa_a_utilizar = masa_helio

def plot_distrib_por_temp(funcion_distrib, temperaturas, m):
    for temp in temperaturas:
        plt.plot(v, 100*funcion_distrib(v, temp, m), label = f'T = {temp} K')
    plt.legend()
    plt.xlabel("Velocidad (m/s)")
    plt.ylabel("Probabilidad porcentual")
    plt.title(f'Ejercicio 3.3.2 \nMasa molar: {m} kg')
    plt.show()

plot_distrib_por_temp(P, temperaturas, Masa_a_utilizar)

""" plt.plot(v, P(v, 298.15, masa_xenon_132), label = f'Xenon-132\nT = {298.15} K')
plt.legend()
plt.show() """

print(f'Ejercicio 3.3.2\nR2: A medida que aumenta la temperatura se puede observar que la velocidad más frecuente o probable aumenta también.')
print(f'Esto puede deducirse de la ecuacion de velocidad promedio, la cual tiene una relacion creciente con la temperatura')
print(f'También, a medida que la temperatura aumenta, la energía cinética también, por lo que la velocidad promedio de las partículas se incrementa.\n')


#3
temperaturas = np.array([50, 100, 200, 300, 500, 700, 1000, 2500, 5000, 10000], dtype='float64')
Masa_a_utilizar = masa_nitrogeno
velocidades_promedio = np.array([])
velocidades_promedio_reales = np.array([])

def velocidad_promedio(m, T):
    M = m/N_a
    def v_pv(x):
        return 4*np.sqrt((k*T)/(2*np.pi*M))*x*np.exp(-x)
    velocidad_promedio_ = Integral_GL_corta(v_pv, 15)
    return velocidad_promedio_

def velocidad_promedio_real(m, T):
    M = m/N_a
    return np.sqrt((8*k*T)/((np.pi)*M))

for temperatura in temperaturas:
    velocidades_promedio = np.append(velocidades_promedio, velocidad_promedio(Masa_a_utilizar, temperatura))
    velocidades_promedio_reales = np.append(velocidades_promedio_reales, velocidad_promedio_real(Masa_a_utilizar, temperatura))

def imprimir_velocidades_promedio(m, v,T):
    print(f'\nEjercicio 3.3.3\nLas velocidades promedio para las siguientes temperaturas con una masa molar de {m} kg son:')
    for temperatura in T:
        i=np.where(T==temperatura)[0][0]
        print(f'T: {temperatura} K --- v={np.round(v[i],4)} m/s') 
        
imprimir_velocidades_promedio(Masa_a_utilizar, velocidades_promedio, temperaturas)

plt.plot(temperaturas, velocidades_promedio, color = 'blue', label = 'Estimado con Gauss-Laguerre', marker = '.')
plt.plot(temperaturas, velocidades_promedio_reales, color =  'green', label = 'Valor real')
plt.legend()
#plt.xscale('log')
plt.xlabel("Temperatura (K)")
plt.ylabel("Velocidad promedio (m/s)")
plt.title(f'Ejercicio 3.3.3\n')
plt.show()


#4
temperaturas = np.array([50, 100, 200, 300, 500, 700, 1000, 2500, 5000, 10000], dtype='float64')
Masa_a_utilizar = masa_nitrogeno
velocidades_media_cuadratica = np.array([])
velocidades_media_cuadratica_reales = np.array([])

def velocidad_media_cuadratica(m, T):
    M = m/N_a
    def v2_pv(x):
        return 4*(k*T/(np.sqrt(np.pi)*M))*x**(1.5)*np.exp(-x)
    velocidad_media_cuadratica_ = Integral_GL_corta(v2_pv, 15)
    return np.sqrt(velocidad_media_cuadratica_)

def velocidad_media_cuadratica_real(m, T):
    M = m/N_a
    return np.sqrt((3*k*T)/(M))

for temperatura in temperaturas:
    velocidades_media_cuadratica = np.append(velocidades_media_cuadratica, velocidad_media_cuadratica(Masa_a_utilizar, temperatura))
    velocidades_media_cuadratica_reales = np.append(velocidades_media_cuadratica_reales, velocidad_media_cuadratica_real(Masa_a_utilizar, temperatura))

def imprimir_velocidades_media_cuadratica(m, v,T):
    print(f'\nEjercicio 3.3.4\nLas velocidades medias cuadráticas para las siguientes temperaturas con una masa molar de {m} kg son:')
    for temperatura in T:
        i=np.where(T==temperatura)[0][0]
        print(f'T: {temperatura} K --- v={np.round(v[i],4)} m/s') 
        
imprimir_velocidades_media_cuadratica(Masa_a_utilizar, velocidades_media_cuadratica, temperaturas)

plt.plot(temperaturas, velocidades_media_cuadratica, color = 'blue', label = 'Estimado con Gauss-Laguerre', marker = '.')
plt.plot(temperaturas, velocidades_media_cuadratica_reales, color =  'green', label = 'Valor real')
plt.legend()
plt.xlabel("Temperatura (K)")
#plt.xscale('log')
plt.ylabel("Velocidad media cuadrática (m/s)")
plt.title(f'Ejercicio 3.3.4')
plt.show()

#5
print(f'\nEjercicio 3.3.5 en pdf adjunto')
    






        