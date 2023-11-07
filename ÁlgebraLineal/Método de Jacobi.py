import numpy as np

A = np.array([[3,-1,-1],[-1.,3.,1.],[2,1,4]])
b = np.array([1.,3.,7.])


def Jacobi(A,b,x0, itmax=1000, tolerancia=1e-9):
    x=x0.copy()
    u=x.copy()
    sumk = x.copy() 
    
    residuo_norma_inf = np.max(np.abs(np.dot(A,x)-b))
    residuo = np.linalg.norm(np.dot(A,x)-b)    
    
    it=0
    
    while residuo >= tolerancia and it < itmax:
        
        u[:] = 0
        sumk[:] = 0
        
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                
                if i!=j:
                    sumk[i] += A[i,j]*x[j]
                
                u[i] = (b[i] - sumk[i])/A[i,i]
        x = u.copy()
        
        residuo_norma_inf = np.max(np.abs(np.dot(A,x)-b))
        residuo = np.linalg.norm(np.dot(A,x)-b)
        
        it+=1
        
        if residuo > 1000:
            print('No calculable con Jacobi')
            x[:] = 0.
            break
    return x

print(Jacobi(A,b,np.array([0.,0.,0.])))
print(A,b)

def GetT(A):
    D = np.zeros_like(A)
    D_inv = np.zeros_like(A)
    R = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i==j:
                D[i,j] = A[i,j]
                D_inv[i,j] = 1/A[i,j]
            else:
                R[i,j] = A[i,j]
    print(D_inv,R)
    T = D_inv @ R
    return T

def GetEig(A):
    return np.linalg.eig(A)

Values, Vectors = GetEig(GetT(A)) 
print(Values)
    
    
    

            