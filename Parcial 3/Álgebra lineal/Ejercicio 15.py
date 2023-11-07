from sympy import *
init_printing(use_unicode=True)
sigma_x = Matrix([[0,1],[1,0]])
sigma_y = Matrix([[0,-I],[I,0]])
sigma_z = Matrix([[1,0],[0,-1]])

def Conmutador(A,B):
    return A*B - B*A

def f(i):
    if i==0:
        return 'x'
    elif i==1:
        return 'y'
    else:
        return 'z'
    
def Operaciones(A,B,C):
    Objetos = [A,B,C]
    for i in range(3):
        for j in range(3):
            print(f'[sigma_{f(i)},sigma_{f(j)}] = {Conmutador(Objetos[i],Objetos[j])}')

print(f'\nCalculado con conmutador\n')
Operaciones(sigma_x, sigma_y, sigma_z)

def Levi_Civita(i,j,k):
    par = [(0,1,2), (2,0,1), (1,2,0)]
    impar = [(2,1,0), (0,2,1), (1,0,2)]
    if (i,j,k) in par:
        return 1
    elif (i,j,k) in impar:
        return -1
    else:
        return 0

            
def ResultadoEsperado(A,B,C):
    matriz_k = Matrix([[0,2,1],[2,1,0],[1,0,2]])
    Objetos = [A,B,C]
    for i in range(3):
        for j in range(3):
            print(f'[sigma_{f(i)},sigma_{f(j)}] = {2*I*Levi_Civita(i,j,matriz_k[i,j])*Objetos[matriz_k[i,j]]}')

print(f'\nEsperado: 2*i*e_ijk*sigma_k\n')
ResultadoEsperado(sigma_x, sigma_y, sigma_z)
print(f'\nEfectivamente, son iguales. Se verifica el álgebra')

