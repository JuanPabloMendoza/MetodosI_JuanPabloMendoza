import numpy as np

matriz = np.matrix([[2,1,1], [1,1,-2], [5,10,5]])
vect = np.array([8,-2,10])


def MatrizAumentada(matriz_coef, solucion):
    n,m=np.shape(matriz_coef)
    
    matriz_aumentada = np.zeros((n,m+1))

    matriz_aumentada[:,0:m] = matriz_coef
    matriz_aumentada[:,m] = solucion
    return matriz_aumentada

def ReducirMatriz(A):
    n,m=np.shape(A)
    print(m,n)
    for j in range(m-2):
        for i in range(j+1,n):
            k = A[i][j]/A[j][j]
            fila_i=A[i]-k*A[j]
            A[i] = fila_i
    return A

def ResolverEcuacion(A,vect):
    #Solo sirve para matrices cuadradas invertibles
    n,m = np.shape(A)
    matriz_aumentada = MatrizAumentada(A,vect)
    matriz_recucida = ReducirMatriz(matriz_aumentada)
    A_transf = matriz_recucida[:,:m]
    vect_transf = matriz_recucida[:,m]
    soluciones = np.zeros(m)
    
    for i in range(n-1,-1,-1):
        solucion_i = vect_transf[i]
        print(list(range(m-1-i,0,-1)))
        max_restar = m-1-i
        j=m-1
        cont=1
        while cont<=max_restar:
            solucion_i -= A_transf[i][j]*soluciones[j]
            cont+=1
            j-=1
        soluciones[i] = solucion_i/A_transf[i][i]
    return soluciones

print(ResolverEcuacion(matriz,vect))
            
            
            

