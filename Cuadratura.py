from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sympy as sym
sym.init_printing(use_unicode=True)



def GetLegendre(n,x,y):
    y=(x**2-1)**n
    poly = sym.diff( y,x,n )/(2**n*np.math.factorial(n))
    return poly

def GetNewton(f,df,xn,itmax=10000,precision=1e-14):
    
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
    
def GetRoots(f,df,x,tolerancia = 14):
    
    Roots = np.array([])
    
    for i in x:
        
        root = GetNewton(f,df,i)
        
        if root != False:
            
            croot = np.round( root, tolerancia )
            
            if croot not in Roots:
                Roots = np.append(Roots, croot)
                
    Roots.sort()
    
    return Roots

x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)

Legendre = []
DLegendre = []
for i in range(7):
    Poly = GetLegendre(i,x,y)
    Legendre.append(Poly)
    DLegendre.append(sym.diff(Poly, x,1))
print(Legendre)

def GetAllRoots(n,xn,Legendre,DLegendre):
    
    poly = sym.lambdify([x],Legendre[n],'numpy')
    Dpoly = sym.lambdify([x],DLegendre[n],'numpy')
    Roots = GetRoots(poly,Dpoly,xn)
    
    return Roots

n = 6
Roots, Weights = np.polynomial.legendre.leggauss(n)
Roots,Weights
a=-2
b=2


def campana_gauss(x):
    return np.exp(-x**2)

t = 0.5*( (b-a)*Roots + a + b )
Integral = 0.5*(b-a)*np.sum(Weights*campana_gauss(t))
print(Integral)




    
