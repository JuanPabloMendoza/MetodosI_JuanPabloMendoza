import numpy as np
import matplotlib.pyplot as plt

def Binomial(n,p,a, N=int(1e4)):
    Evento = np.random.binomial(n,p,N)
    Pedido = Evento[Evento<=a]
    return Pedido.size/N

def Poisson1(lam,a,N=int(1e4)):
    Eventos = np.random.poisson(lam,N)
    Pedido = Eventos[Eventos==a]
    return Pedido.size/N
def Poisson2(lam,a,N = int(1e4)):
    Eventos = np.random.poisson(lam,N)
    Pedido = Eventos[Eventos>a]
    return Pedido.size/N

def GraphBin(Caso,N=int(1e3)):
    P = np.linspace(0,1,N)
    for caso in Caso: 
        Probabilidad = np.empty((N))
        for i in range(N):
            Probabilidad[i] = Binomial(caso[0],P[i],caso[1])
        plt.plot(P,Probabilidad, label=f'N={caso[0]}, a={caso[1]}')
    plt.ylabel('Probabilidad de aceptación')
    plt.xlabel('p')
    plt.legend()
    plt.show()
    
def GraphPoisson(lam ,N=10):
    a = np.array(list(range(N)))
    Probabilidad = np.empty((N))
    for i in range(N):
        Probabilidad[i] = Poisson1(lam, i)
    plt.plot(a, Probabilidad)
    plt.ylabel('Probabilidad')
    plt.xlabel('Num.')
    plt.show()

Caso = np.array([[5,1],[25,5],[100,30]])
GraphBin(Caso)


lam = 5
GraphPoisson(lam)
prob = 0
for i in range(0,3):
    prob+=Poisson1(lam,i)
print(prob)

print(Poisson2(lam, 10))

