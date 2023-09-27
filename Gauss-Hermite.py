import numpy as np
import sympy as sym

x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)

#GAUSS-HERMITE
def GetHermiteRecursive(n,x):

    if n==0:
        poly = sym.Pow(1,1)
    elif n==1:
        poly = 2*x
    else:
        poly = 2*x*GetHermiteRecursive(n-1, x) - sym.diff(GetHermiteRecursive(n-1, x), x, 1)
    return sym.simplify(poly)
n=4
print(f'\nEjercicio 3.2.1\nPolinomio de Hermite de grado {n}: {GetHermiteRecursive(n,x)}\n')

#RAICES

def GetDHermite(n,x):
    Pn = GetHermiteRecursive(n,x)
    return sym.diff(Pn,x,1)

def GetNewton(f,df,xn,itmax=10000,precision=1e-10):
    
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
    
def GetRootsGHer(f,df,x,tolerancia = 12):
    
    Roots = np.array([])
    
    for i in x:
        
        root = GetNewton(f,df,i)
        croot = np.round( root, tolerancia )
        
        if croot not in Roots:
            Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots

def GetAllRootsGHer(n):
    a = -np.sqrt(4*n+1)
    b = np.sqrt(4*n+1)
    xn = np.linspace(a,b,100)
    
    Hermite = []
    DHermite = []
    
    for i in range(n+1):
        Hermite.append(GetHermiteRecursive(i,x))
        DHermite.append(GetDHermite(i,x))
    
    poly = sym.lambdify([x],Hermite[n],'numpy')
    Dpoly = sym.lambdify([x],DHermite[n],'numpy')
    Roots = GetRootsGHer(poly,Dpoly,xn)
    
    return Roots

n=4
print(f'Ejercicio 3.2.2\nRaices del polinomio de Hermite de grado {n}: {GetAllRootsGHer(n)}\n')


def GetWeightsGHer(n):

    Roots = GetAllRootsGHer(n)

    

    Hermite = []
    
    for i in range(n+1):
        Hermite.append(GetHermiteRecursive(i,x))
    
    poly = sym.lambdify([x],Hermite[n-1],'numpy')
    Weights = []
    for root in Roots:
        Weights.append(np.round((2**(n-1))*np.math.factorial(n)*np.sqrt(np.pi) / ((n**2) * poly(root)**2), 8))
    
    return Weights

n=4
print(f'Ejercicio 3.2.3\nPesos del polinomio de Hermite de grado {n}: {GetWeightsGHer(n)}\n')


#INTEGRAL -INF A INF

def Integral_GH(funcion, n):
    def funcion_corregida(x):
        return funcion(x) * np.exp(x**2)
    raices = GetAllRootsGHer(n)
    pesos = GetWeightsGHer(n)
    suma = 0
    for i in range(n):
        suma += pesos[i] * funcion_corregida(raices[i])
    return suma

def Integral_GH_corta(funcion, n):
    # Más rápida. Usa los pesos y raices dados por numpy
    def funcion_corregida(x):
        return funcion(x) * np.exp(x**2)
    raices,pesos = np.polynomial.hermite.hermgauss(n)
    suma = 0
    for i in range(n):
        suma += pesos[i] * funcion_corregida(raices[i])
    return suma

def funct(x):
    return np.exp(-(x**2))

print(f'Ejemplo con la función e^(-(x^2)):')
print(Integral_GH(funct, 6))
#print(Integral_GH_corta(funct, 6))




    

