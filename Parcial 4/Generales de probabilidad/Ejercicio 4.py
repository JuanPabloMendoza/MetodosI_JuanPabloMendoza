import numpy as np
import matplotlib.pyplot as plt
#formula general: P(n) = productoria desde i=0 hasta i=n de (365-i)/365. Si el año es bisiesto, se reemplaza 365 por 366.
#para n>365 (o 366), la probabilidad es 1
def Cumpleaños(n,dias=365):
    if n<dias:
        prob = 1
        for i in range(n):
            prob *= (dias-i)/dias
        return prob
    else:
        return 1

n=50
print(f'Ejemplo: La probabilidad de que de {n} personas todas ellas cumplan años en días distintos es de {np.round(Cumpleaños(n),5)*100}%')

n_=80
x = np.arange(1,n_)
y = np.empty((x.size))
for i in range(x.size):
    y[i] = Cumpleaños(x[i])
plt.plot(x,y)
plt.title(f'Probabilidad de que n personas cumplan en días distintos.')
plt.xlabel(f'n')
plt.ylabel(f'P(n)')
plt.show()
