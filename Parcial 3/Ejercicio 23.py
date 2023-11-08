import numpy as np

A = np.array([[0.2,0.1,1,1,0],[0.1,4,-1,1,-1],[1,-1,60,0,-2],[1,1,0,8,4],[0,-1,-2,4,700]], dtype='float64')
b = np.array([1,2,3,4,5],dtype='float64')

def DescensoConjugado(A,b, x0, tolerancia=0.01):
    r0 = np.dot(A,x0)-b
    p0 = -r0
    k=0
    p=p0
    r=r0
    x=x0
    while np.max(np.abs(r))>tolerancia:
        alpha = -(np.dot(r.T, p)/np.dot(np.dot(p.T, A),p))
        x = x + np.dot(alpha, p)
        r = np.dot(A,x) - b
        beta = (np.dot(np.dot(r.T, A), p)/np.dot(np.dot(p.T, A),p))
        p = -r + np.dot(beta, p)
        k=k+1
    print(f'Converge en {k} iteraciones.')
    return x

seed = np.array([0.,0.,0.,0.,0.], dtype='float64')  
sol = DescensoConjugado( A, b, seed)
print(f'La solución al sistema es: \n{sol}\n')
print(f'{A}\n*\n{sol.T}\n=\n{np.dot(A,sol)}\ncomo queriamos.')
    
    
    

