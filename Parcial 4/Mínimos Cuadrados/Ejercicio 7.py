import numpy as np
#a

A = np.array([[3,1,0,1],[1,2,1,1],[-1,0,2,-1]]).T
b = np.array([-3,-3,8,9])
AT = A.T
M = np.dot(AT,A)
bT = np.dot(AT,b)
x = np.linalg.solve(M,bT)
print(f'Solución por mínimos cuadrados: {np.round(np.dot(A,x),5)}')


#b
#tomé parte del código de https://gist.github.com/iizukak/1287876/edad3c337844fac34f7e56ec09f9cb27d4907cc7
def GetGramSchmidt(Vectors):
    B = []
    for v in Vectors:
        w = v - sum(np.dot(v,b)*b  for b in B)
        #np.sum manda un warning, no se por qué.
        if (w > 1e-8).any():  
            B.append(w/np.linalg.norm(w))
    return np.array(B)

b = np.array([-3,-3,8,9])
Vectores = np.array([[3,1,0,1],[1,2,1,1],[-1,0,2,-1]])
OrtNormBasis = GetGramSchmidt(Vectores)

def GetProjection(OrtNormBasis,vector):
    orth = 0
    for v in OrtNormBasis:
        orth += np.dot(v,vector)*v
    return np.round(orth,6)
R = GetProjection(OrtNormBasis, b)
print(f'Solución por Grand-Schmidt: {R}')