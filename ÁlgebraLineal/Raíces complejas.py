import numpy as np
import sympy as sym

x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)


def Jn(f,r,h=1e-6):
    
    n = r.shape[0]
    
    J = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            
            rf = r.copy()
            rb = r.copy()
            
            rf[j] = rf[j] + h
            rb[j] = rb[j] - h
            
            J[i,j] = ( f[i](rf[0],rf[1]) - f[i](rb[0],rb[1])  )/(2*h)
            
    return J

def f(x,y):
    
    z = x+sym.I*y
    f = z**2 + 1
    f = f.expand()
    return sym.re(f), sym.im(f)

f0, f1 = f(x,y)

F = [f0,f1]

J = sym.zeros(2,2)

for i in range(2):
    for j in range(2):
        if j==0:
            J[i,j] = sym.diff(F[i],x,1)
        else:
            J[i,j] = sym.diff(F[i],y,1)
            
InvJ = J.inv(method='LU')
print(InvJ)

Fn = sym.lambdify([x,y],F,'numpy')
IJn = sym.lambdify([x,y],InvJ,'numpy')

print(IJn(1.,1.))

def NewtonRaphson(z,Fn,Jn,itmax=100,precision=1e-9):
    
    error = 1
    it = 0
    
    while error > precision and it < itmax:
        
        IFn = Fn(z[0],z[1])
        IJn = Jn(z[0],z[1])
        
        z1 = z - np.dot(IJn,IFn)
        
        error = np.max( np.abs(z1-z) )
        
        z = z1
        it +=1
        
    print(it)
    return z
print(NewtonRaphson([1,1], Fn, IJn))