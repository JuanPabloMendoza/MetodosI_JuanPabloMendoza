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

A = np.array([[1,0,0],[5,1,0],[-2,3,1]], dtype='float64')
B = np.array([[4,-2,1],[0,3,7],[0,0,2]], dtype='float64')

print(f'AB =\n{MaxtrixMultiplication(A,B)}')
