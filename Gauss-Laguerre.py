import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)


#GAUSS-LAGUERRE
def GetLaguerreRecursive(n,x):
    if n==0:
        poly = sym.Pow(1,1)
    elif n==1:
        poly = 1-x
    else:
        poly = ((2*n-1-x)*GetLaguerreRecursive(n-1,x)-(n-1)*GetLaguerreRecursive(n-2,x))/n
   
    return sym.simplify(poly)

def GetLaguerreDirect(n, x):
    Laguerre = [sym.Pow(1,1), 1-x]
    for i in range(2,n+2):
        Poly = ((2*i-1-x)*Laguerre[i-1]-(i-1)*Laguerre[i-2])/i
        Laguerre.append(sym.simplify(Poly))
    return Laguerre

#RAICES

def GetDLaguerre(Poly,x):
    return sym.diff(Poly,x,1)

def GetNewton(f,df,xn,itmax=100000,precision=1e-10):
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

def GetRootsLag(x, newtonmethod, f, df):
    roots = np.array([], dtype='float64')
    for i in range(x.size):
        x_ = x[i]
        root = newtonmethod(f, df, x_)
        roots = np.append(roots, root)
    return roots
 
def SkipRoots(roots, f=9, e=0.0001):
    roots = np.sort(roots)
    real_roots = np.array([], dtype='float64')
    root_i = roots[0]
    real_roots = np.append(real_roots, np.round(root_i,f))
    for i in range(roots.size):
        difference = root_i-roots[i]
        if not np.abs(difference)<e and (np.round(root_i,f) not in real_roots):
            real_roots = np.append(real_roots, np.round(root_i,f))
        root_i= roots[i]
    if np.round(root_i,f) not in real_roots:
        real_roots = np.append(real_roots, np.round(root_i,f))
    return np.sort(real_roots)

def GetAllRootsGLag(Laguerre, DLaguerre, n):
    a = 0
    b = n+(n-1)*np.sqrt(n)
    xn = np.linspace(a,b,300)
    
    poly = sym.lambdify([x],Laguerre[n],'numpy')
    Dpoly = sym.lambdify([x],DLaguerre[n],'numpy')
    Roots = SkipRoots(GetRootsLag(xn, GetNewton, poly,Dpoly))
    
    return Roots

def GetWeightsGLag(Laguerre, Roots, n):
    poly = sym.lambdify([x],Laguerre[n+1],'numpy')
    Weights = []
    for root in Roots:
        Weights.append(np.round(root/(((n+1)**2)*(poly(root))**2), 8))
    
    return Weights

def GetEverythingLaguerre(n):
    
    Laguerre = GetLaguerreDirect(n, x)
    DLaguerre = []
    for poly in Laguerre:
        DLaguerre.append(GetDLaguerre(poly, x))
        
    Roots = GetAllRootsGLag(Laguerre, DLaguerre, n)
    Weights = GetWeightsGLag(Laguerre, Roots, n)
   
    return Laguerre[n], Roots, Weights



#INTEGRAL 0 A INF

def Integral_GL(funcion, n):
    def funcion_corregida(x):
        return funcion(x) * np.exp(x)
    Laguerre, Roots, Weights = GetEverythingLaguerre(n)
    suma = 0
    for i in range(n):
        suma += Weights[i] * funcion_corregida(Roots[i])
    return suma

def Integral_GL_corta(funcion, n):
    # Más rápida. Usa los pesos y raices dados por numpy
    def funcion_corregida(x):
        return funcion(x) * np.exp(x)
    Roots,Weights = np.polynomial.laguerre.laggauss(n)
    suma = 0
    for i in range(n):
        suma += Weights[i] * funcion_corregida(Roots[i])
    return suma

def fun(x):
    #no funciona con esta jajaja
    return np.sin(x)/x


X = np.linspace(-50,50, 10000000)
Y = fun(X)
plt.plot(X,Y)

plt.show()
n= 101
N = np.arange(2,n)
Int = np.array([])
for i in range(2, n):
    Int = np.append(Int, Integral_GL_corta(fun, i))
plt.scatter(N, Int)
plt.xlabel('N')
plt.ylabel('Integral Si(x)')
plt.grid()
plt.show()






        