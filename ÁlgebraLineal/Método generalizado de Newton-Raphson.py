import numpy as np
import sympy as sym

G=(lambda x,y,z: 6*x - 2*np.cos(y*z) - 1., \
    lambda x,y,z: 9*y + np.sqrt((x**2) + np.sin(z)+ 1.06) + 0.9, \
    lambda x,y,z: 60*z + 3*np.exp((-x)*y) + 10*np.pi -3)

G=(lambda x,y: x**2 - y**2 + 1., \
    lambda x,y: 2*x*y)

""" def GetF(G,r):
    
    n = r.shape[0]
    
    v = np.zeros_like(r)
    
    for i in range(n):
        v[i] = G[i](r[0],r[1],r[2])
        
    return v """

def GetF(G,r):
    
    n = r.shape[0]
    
    v = np.zeros_like(r)
    
    for i in range(n):
        v[i] = G[i](r[0],r[1])
        
    return v

def Metric(G,r):
    return 0.5*np.linalg.norm(GetF(G,r))**2

""" def GetJacobian(f,r,h=1e-6):
    
    n = r.shape[0]
    
    J = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            
            rf = r.copy()
            rb = r.copy()
            
            rf[j] = rf[j] + h
            rb[j] = rb[j] - h
            
            J[i,j] = ( f[i](rf[0],rf[1],rf[2]) - f[i](rb[0],rb[1],rb[2])  )/(2*h)
            
    return J """

def GetJacobian(f,r,h=1e-6):
    
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

def NewtonRaphson(G,r,itmax=100000, error=1e-9):
    
    it = 0
    d = 1.
    dvector = []
    
    while d>error and it<itmax:
        
        # Vector actual
        rc = r
        
        F = GetF(G,rc)
        J = GetJacobian(G, rc)
        InvJ = np.linalg.inv(J)
        
        r = rc - np.dot(InvJ, F)
        
        diff = r - rc
        
        d = np.max( np.abs(diff))
        
        dvector.append(d)
        it+=1
    print(it)
    return r, dvector


r, dvector = NewtonRaphson(G, np.array([-1., -1.]))
print(r)


