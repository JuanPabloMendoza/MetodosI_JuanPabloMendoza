import numpy as np
#Solo se puede encontrar el valor propio máximo o mínimo

def MaxtrixMultiplication(A,B):
    #Check dim
    a,b = A.shape
    c,d = B.shape
    if b!=c:
        return None
    n=c
    AB = np.zeros(shape=(a,d))
    for i in range(a):
        for j in range(d):
            for k in range(n):
                AB[i,j] += A[i,k]*B[k,j]
    return AB

z = np.matrix([[-1],[-1],[-1]], dtype='float64')

def RayleighQuotient(A,v):
    return (MaxtrixMultiplication(v.T, MaxtrixMultiplication(A,v)))/(MaxtrixMultiplication(v.T, v))

def MaxEigenValue(A, z, tolerancia=1e-7, itmax=1000):
    Normalized = (1/np.linalg.norm(z))*z
    EValue = 0
    Error = np.linalg.norm(MaxtrixMultiplication(A,Normalized) - EValue*Normalized)
    it=0
    while Error>tolerancia and itmax>it:
        z = MaxtrixMultiplication(A,Normalized)
        Normalized = (1/np.linalg.norm(z))*z
        EValue = RayleighQuotient(A,Normalized)
        Error = np.linalg.norm(MaxtrixMultiplication(A,Normalized) - EValue*Normalized)
        it+=1
    return EValue[0,0], Normalized

def MinEigenValue(A, z, tolerancia=1e-7, itmax=1000):
    return MaxEigenValue(np.linalg.inv(A), z, tolerancia, itmax)

H = np.matrix([[1,2,-1],[1,0,1],[4,-4,5]], dtype='float64')
autovalor, autovector = MinEigenValue(H,z)
print(f'Eo = {np.round(autovalor,4)}, \n|psi 0> = \n{autovector}')

