import numpy as np


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

Markov = np.array([[0.6, 0.8], [0.4, 0.2]], dtype='float64')
res = Markov
N=100
for i in range(N-1):
    res = np.round(MaxtrixMultiplication(res, Markov),2)
    print(res)
    


