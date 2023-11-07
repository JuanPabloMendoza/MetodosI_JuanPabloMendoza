import numpy as np


#Solo se puede encontrar el valor propio máximo o mínimo

def RayleighQuotient(A,v):
    return (np.dot(v.T, np.dot(A,v)))/(np.dot(v.T, v))

def MaxEigenValue(A, z, tolerancia=1e-7, itmax=1000):
    Normalized = (1/np.linalg.norm(z))*z
    EValue = 0
    Error = np.linalg.norm(np.dot(A,Normalized) - EValue*Normalized)
    it=0
    while Error>tolerancia and itmax>it:
        z = np.dot(A,Normalized)
        Normalized = (1/np.linalg.norm(z))*z
        EValue = RayleighQuotient(A,Normalized)
        Error = np.linalg.norm(np.dot(A,Normalized) - EValue*Normalized)
        it+=1
    return EValue, Normalized, it

def MinEigenValue(A, z, tolerancia=1e-7, itmax=1000):
    A = np.linalg.inv(A)
    Normalized = (1/np.linalg.norm(z))*z
    EValue = 0
    Error = np.linalg.norm(np.dot(A,Normalized) - EValue*Normalized)
    it=0
    while Error>tolerancia and itmax>it:
        z = np.dot(A,Normalized)
        Normalized = (1/np.linalg.norm(z))*z
        EValue = RayleighQuotient(A,Normalized)
        Error = np.linalg.norm(np.dot(A,Normalized) - EValue*Normalized)
        it+=1
    return (1/EValue), Normalized, it

A = np.array([[-5,2], [-7,4]], dtype='float64')
z = np.array([1,0])

print(MaxEigenValue(A,z))
print(MinEigenValue(A,z))


#E2 from Ex.11
A = np.array([[1,2,-1], [1,0,1], [4,-4,5]], dtype='float64')
z = np.array([1,0,0])
print(MinEigenValue(A,z))


        