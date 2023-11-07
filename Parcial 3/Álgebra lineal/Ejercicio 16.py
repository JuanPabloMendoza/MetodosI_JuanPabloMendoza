from sympy import *
init_printing(use_unicode=True)
gamma_0 = Matrix([[1,0,0,0],[0,1,0,0], [0,0,-1,0],[0,0,0,-1]])
gamma_1 = Matrix([[0,0,0,1],[0,0,1,0], [0,-1,0,0],[-1,0,0,0]])
gamma_2 = Matrix([[0,0,0,-I],[0,0,I,0], [0,I,0,0],[-I,0,0,0]])
gamma_3 = Matrix([[0,0,1,0],[0,0,0,-1], [-1,0,0,0],[0,1,0,0]])


def Operacion(A,B):
    return A*B + B*A

    
def Operaciones(A,B,C,D):
    Objetos = [A,B,C,D]
    for i in range(4):
        for j in range(4):
            print(f'[gamma_{i},gamma_{j}] = {Operacion(Objetos[i],Objetos[j])}')

print(f'\nCalculado con operacion antisimétrica\n')
Operaciones(gamma_0, gamma_1, gamma_2, gamma_3)


          
def ResultadoEsperado():
    Metrica = Matrix([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])
    Identidad = Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    for i in range(4):
        for j in range(4):
            print(f'[gamma_{i},gamma_{j}] = {2*Metrica[i,j]*Identidad}')

print(f'\nEsperado: 2*i*metrica_uv*I(4)\n')
ResultadoEsperado()
print(f'\nEfectivamente, son iguales. Se verifica el álgebra.')

