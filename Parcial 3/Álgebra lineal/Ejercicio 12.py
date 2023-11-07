import numpy as np


#Metodo Generalizado de Newton-Raphson

def GetF3(G,r):
    
    n = r.shape[0]
    
    v = np.zeros_like(r)
    
    for i in range(n):
        v[i] = G[i](r[0],r[1],r[2])
        
    return v 

def GetF2(G,r):
    
    n = r.shape[0]
    
    v = np.zeros_like(r)
    
    for i in range(n):
        v[i] = G[i](r[0],r[1])
        
    return v

def GetF(G, r):
    if len(G)==2:
        return GetF2(G,r)
    if len(G)==3:
        return GetF3(G,r)

def Metric3(G,r):
    return 0.5*np.linalg.norm(GetF3(G,r))**2
def Metric2(G,r):
    return 0.5*np.linalg.norm(GetF2(G,r))**2

def GetMetric(G, r):
    if len(G)==2:
        return Metric2(G,r)
    if len(G)==3:
        return Metric3(G,r)

def GetJacobian3(f,r,h=1e-6):
    
    n = r.shape[0]
    
    J = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            
            rf = r.copy()
            rb = r.copy()
            
            rf[j] = rf[j] + h
            rb[j] = rb[j] - h
            
            J[i,j] = ( f[i](rf[0],rf[1],rf[2]) - f[i](rb[0],rb[1],rb[2]))/(2*h)
            
    return J

def GetJacobian2(f,r,h=1e-6):
    
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

def GetJacobian(G, r):
    if len(G)==2:
        return GetJacobian2(G,r)
    if len(G)==3:
        return GetJacobian3(G,r)
    
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
    return r, dvector

#Descenso al gradiente

def Descenso3D(f,seed,N,gamma = 0.00001):
    Dx = lambda f,x,y,z,h=1e-5: (f(x+h,y,z) - f(x-h,y,z))/(2*h)
    Dy = lambda f,x,y,z,h=1e-5: (f(x,y+h,z) - f(x,y-h,z))/(2*h)
    Dz = lambda f,x,y,z,h=1e-5: (f(x,y,z+h) - f(x,y,z-h))/(2*h)

    Gradient = lambda f,x,y,z: np.array([Dx(f,x,y,z),Dy(f,x,y,z),Dz(f,x,y,z)])
    #Minimizer
    
    r = np.zeros((N,3))
    r[0] = seed
    
    Grad = np.zeros((N,3))
    Grad[0] = Gradient(f,r[0,0],r[0,1],r[0,2])
    

    for i in range(1,N):
        r[i] = r[i-1] - gamma*Gradient(f,r[i-1,0],r[i-1,1],r[i-1,2])
        Grad[i] = Gradient(f,r[i-1,0],r[i-1,1],r[i-1,2])
        
    return r[N-1],Grad

def Descenso2D(f,seed,N,gamma = 0.01):
    Dx = lambda f,x,y,h=1e-5: (f(x+h,y) - f(x-h,y))/(2*h)
    Dy = lambda f,x,y,h=1e-5: (f(x,y+h) - f(x,y-h))/(2*h)
    
    Gradient = lambda f,x,y: np.array([Dx(f,x,y),Dy(f,x,y)])
    #Minimizer
    
    r = np.zeros((N,2))
    r[0] = seed
    
    Grad = np.zeros((N,2))
    Grad[0] = Gradient(f,r[0,0],r[0,1])

    for i in range(1,N):
        r[i] = r[i-1] - gamma*Gradient(f,r[i-1,0],r[i-1,1])
        Grad[i] = Gradient(f,r[i-1,0],r[i-1,1])
        
    return r[N-1],Grad

#Newton-Raphson
print(f'\nNewton-Raphson\n')

#1
G=(lambda x,y: np.log(x**2 + y**2) - np.sin(x*y) - np.log(2*np.pi), \
    lambda x,y: np.exp(x-y) + np.cos(x*y))
r, dvector = NewtonRaphson(G, np.array([2.,2.]))
print(f'1: {r}')
    
#2
G=(lambda x,y,z: 6*x-2*np.cos(y*z)-1, \
    lambda x,y,z: 9*y + np.sqrt((x**2) + np.sin(z)+ 1.06) + 0.9, \
    lambda x,y,z: 60*z + 3*np.exp((-x)*y) + 10*np.pi -3)
r, dvector = NewtonRaphson(G, np.array([0.,0.,0.]))
print(f'2: {r}')

#Descenso al gradiente
print(f'\nDescenso al gradiente\n')
#1
f = lambda x,y: (np.log(x**2 + y**2) - np.sin(x*y) - np.log(2*np.pi))**2 + (np.exp(x-y) + np.cos(x*y))**2
N = 200
r,Grad = Descenso2D(f,np.array([2.,2.]),N)
print(f'1: {r}')
#2
f = lambda x,y,z: (6*x-2*np.cos(y*z)-1)**2 + (9*y + np.sqrt((x**2) + np.sin(z)+ 1.06) + 0.9)**2 + (60*z + 3*np.exp((-x)*y) + 10*np.pi -3)**2
N = 20000
r,Grad = Descenso3D(f,np.array([0.,0.,0.]),N)
print(f'2: {r}')
