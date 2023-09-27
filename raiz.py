import numpy as np
import matplotlib.pyplot as plt
n=10
func = np.zeros(n)
type(func)


def raiz_cuadrada(x, fn_1_x):
    return ((x/fn_1_x)+fn_1_x)/2

def raiz_cuadrada_definitiva(x,n):
    func = np.zeros(n)
    x=2
    func[0]=1
    for i in range(1,n):
        func[i]=raiz_cuadrada(x, func[i-1])
    return func[n-1]


k=np.arange(0,n,1)
print(k, func)
plt.scatter(k, func)
plt.axhline(y=np.sqrt(2), color='r')


def funcion(x):
    return 3*x+5

def distancia(xi, funcion, incremento, P):
    interior_i = ((xi-P[0])^2)+((funcion(xi)-P[1])^2)
    xi+=incremento
    interior_f = ((xi-P[0])^2)+((funcion(xi)-P[1])^2)
    while interior_f<interior_i:
        interior_i=interior_f
        xi+=incremento
        interior_f = ((xi-P[0])^2)+((funcion(xi)-P[1])^2)
    return raiz_cuadrada_definitiva(interior_i, 100)

print(distancia(-5, funcion, 0.25, [0,0]))

    
    