import numpy as np
import sympy as sym

x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)

#Newton-Raphson
def GetNewtonMethod(f,df,xn,itmax=10000,precision=1e-10):
    
    error = 1.
    it=0
    
    while error > precision and it<itmax:
        try:
            xn1 = xn - f(xn)/df(xn)
            error = np.abs(f(xn)/df(xn))
        except ZeroDivisionError:
            print('Division por cero')
        xn=xn1
        it+=1
    return xn

def GetRoots(x, newtonmethod, f, df):
    roots = np.array([], dtype='float64')
    for i in range(X.size):
        x_ = x[i]
        root = newtonmethod(f, df, x_)
        roots = np.append(roots, root)
    return roots

def SkipRoots(roots, f=7, e=0.01):
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

#Laguerre 

def GetLaguerre(x, n):
    f = sym.exp(-x)*x**n
    f2 = sym.exp(x)/sym.factorial(n)
    return sym.simplify(f2*sym.diff(f, x, n))

def sympy_diff(x, fun):
    return sym.simplify(sym.diff(fun, x, 1))

n=30
#Cota dada en el taller 2
b=n+(n-1)*np.sqrt(n)
X= np.linspace(0,b, 600)
poly = GetLaguerre(x, n)
poly_f = sym.lambdify([x],poly,'numpy')
Dpoly = sympy_diff(x, poly)
Dpoly_f = sym.lambdify([x],Dpoly,'numpy') 

Raices = SkipRoots(GetRoots(X, GetNewtonMethod, poly_f, Dpoly_f), f=9)
print(Raices)
print(np.size(Raices))


