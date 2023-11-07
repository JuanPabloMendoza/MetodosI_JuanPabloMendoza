import numpy as np

#----------
import sys
import os
def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper
#----------
A = np.array([[3,-1,-1],[-1.,3.,1.],[2,1,4]])
b = np.array([1.,3.,7.])


def GetSOR(A, b, x0, w=1.8, error=1e-10, itmax=1e4):
    n,m = A.shape
    D = np.zeros(shape=(n,m))
    U = np.zeros(shape=(n,m))
    L = np.zeros(shape=(n,m))
    if n!=m:
        print(f'La matriz debe ser cuadrada')
        return None, None
    for i in range(n):
        for j in range(m):
            if i>j:
                L[i,j] = A[i,j]
            elif i==j:
                D[i,j] = A[i,j]
            else:
                U[i,j] = A[i,j]
    x=x0
    res = np.linalg.norm(np.dot(A,x)-b)
    it = 0
    while res > error and it<itmax:
        x_new = x.copy()
        for i in range(n):
            sum_i = (1-w)*x[i] + (w/D[i,i])*(b[i]-np.dot(U,x)[i]-np.dot(L,x)[i])
            if sum_i!= float('inf'):
                x_new[i] = sum_i
            else:
                #print(f'No converge.')
                return 'No converge.', 'No converge.'
        x=x_new     
        res = np.linalg.norm(np.dot(A,x_new)-b)
        it+=1
    if it==itmax:
        print(f'No converge en {itmax} iteraciones con w={w}')
        return None, itmax
    return x, it

print(GetSOR(A,b,np.array([0.,0.,0.]), w=1.0141253912695636, error=1e-4))


#mediante experimentos (puede calcularse también con el radio espectral) encontré que para w>1.64489 aprox, el método no converge.
w_max=1.64489
@blockPrinting
def GetWMin(A, b, x0, error=1e-4, N=20000):
    W = np.linspace(0,w_max,N)
    incert=w_max/N
    #claramente para 0 no convergerá
    lista_it_w = {}
    i=0
    for w in W:
        x, it = GetSOR(A,b,x0,w, error)
        """ if it<=it_min:
            it_min = it
            w_min = w """
        if it not in lista_it_w:
            lista_it_w[it]=np.array([w])
        else:
            lista_it_w[it]= np.append(lista_it_w[it],w)
        i+=1
        print(i)
        
    return lista_it_w[sorted(lista_it_w)[0]], sorted(lista_it_w)[0], incert


err=1e-8
ws,it_min, incert = GetWMin(A,b,np.array([0.,0.,0.]), err)
print(f'Con un error máximo de {err} para la solución, el mínimo de iteraciones es {it_min} y los parámetros de relajación que requieren este número de iteraciones están en un rango entre w={ws[0]} +- {incert} y w={ws[ws.size-1]} +- {incert} aproximadamente.')
print(f'Por ejemplo, w={ws[int((ws.size-1)/2)]}')
