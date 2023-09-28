import numpy as np
import sympy as sym

x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)

#RAICES

def GetNewton(f,df,xn,itmax=10000000,precision=1e-13):
    
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
    
def GetRootsGHer(f,df,x,tolerancia = 11):
    
    Roots = np.array([])
    
    for i in x:
        
        root = GetNewton(f,df,i)
        croot = np.round( root, tolerancia )
        
        if croot not in Roots:
            Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots

#GAUSS-HERMITE

def GetHermiteRecursive(n,x):

    if n==0:
        poly = sym.Pow(1,1)
    elif n==1:
        poly = 2*x
    else:
        poly = 2*x*GetHermiteRecursive(n-1, x) - sym.diff(GetHermiteRecursive(n-1, x), x, 1)
    return sym.simplify(poly)

def GetHermiteDirect(n, x):
    Hermite = []
    #Hermite.append(2*x)
    for i in range(0,n+1):
        Poly = ((-1)**i)*(sym.exp(x**2))*sym.diff(sym.exp(-(x**2)), x, i)
        Hermite.append(Poly)
    return Hermite

def GetDHermite(Poly,x):
    return sym.diff(Poly,x,1)

def GetAllRootsGHer(Hermite, DHermite, n):
    a = -np.sqrt(4*n+1)
    b = np.sqrt(4*n+1)
    xn = np.linspace(a,b,100)
    
    poly = sym.lambdify([x],Hermite[n],'numpy')
    Dpoly = sym.lambdify([x],DHermite[n],'numpy')
    Roots = GetRootsGHer(poly,Dpoly,xn)
    
    return Roots

def GetWeightsGHer(Hermite, Roots, n):
    poly = sym.lambdify([x],Hermite[n-1],'numpy')
    Weights = []
    for root in Roots:
        Weights.append(np.round((2**(n-1))*np.math.factorial(n)*np.sqrt(np.pi) / ((n**2) * poly(root)**2), 8))
    
    return Weights

def GetEverythingHermite(n):
    
    Hermite = GetHermiteDirect(n, x)
    DHermite = []
    for poly in Hermite:
        DHermite.append(GetDHermite(poly, x))
        
    Roots = GetAllRootsGHer(Hermite, DHermite, n)
    Weights = GetWeightsGHer(Hermite, Roots, n)
   
    return Hermite[n], Roots, Weights

#INTEGRAL -INF A INF

def Integral_GH(funcion, n):
    def funcion_corregida(x):
        return funcion(x) * np.exp(x**2)
    Hermite, Roots, Weights = GetEverythingHermite(n)
    suma = 0
    for i in range(n):
        suma += Weights[i] * funcion_corregida(Roots[i])
    return suma

def Integral_GH_rapida(funcion, n):
    # Más rápida. Usa los pesos y raices dados por numpy
    def funcion_corregida(x):
        return funcion(x) * np.exp(x**2)
    raices,pesos = np.polynomial.hermite.hermgauss(n)
    print(raices, pesos)
    suma = 0
    for i in range(n):
        suma += pesos[i] * funcion_corregida(raices[i])
    return suma

#a
n=20

Poly, Roots, Weights = GetEverythingHermite(n)
print(f'\nPolinomio de Hermite de grado {n}: {Poly}\n')
print(f'\nRaices del polinomio de Hermite de grado {n}: {Roots}\n')
print(f'\nPesos del polinomio de Hermite de grado {n}: {Weights}\n')


#b
def oscilador_armonico(x, n=1):
    Hermite = GetHermiteDirect(n, sym.Symbol('x',real=True))
    Hermite_fun = sym.lambdify([sym.Symbol('x',real=True)],Hermite[n],'numpy')
    return (1/np.sqrt((2**n)*np.math.factorial(n)))*np.sqrt(np.sqrt(1/np.pi))*np.exp(-((x**2)/2))*Hermite_fun(x)


def func(x):
    return (x**2)*(oscilador_armonico(x)**2)

print(Integral_GH_rapida(func, 30))
print(Integral_GH(func, 30))




    

