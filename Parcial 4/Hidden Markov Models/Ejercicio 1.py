import numpy as np
from itertools import permutations
from itertools import combinations
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt

def GetStates(States,N):
    
    CStates = list( combinations_with_replacement(States,N) )
    
    Permu = []
    
    for it in CStates:
        p = list(permutations(it,N))
        
        for i in p:
            if i not in Permu:
                Permu.append(i)
                
    return np.array(Permu)

def GetProbObs(T,State,Prior):
    
    n = len(State)
    p = 1.
    
    p *= Prior[State[0]]
    
    # Matriz de transicion
    for i in range(n-1):
        p *= T[ State[i+1], State[i]  ]
        
    return p

def GetProb(T,E,Obs,State,Prior):
    
    n = len(Obs)
    p = 1.
    
    p *= Prior[State[0]]
    
    # Matriz de transicion
    for i in range(n-1):
        p *= T[ State[i+1], State[i]  ]
        
    for i in range(n):
        p *= E[ Obs[i], State[i] ]
        
    return p

def GetS(S, cod):
    Sec = []
    for i in range(S.size):
        x = int(S[i])
        Sec.append(cod[x])
    return Sec

def GetHiddenStatesInfo(HEvents, N, Obs, T, E, Prior, DictO, graph=True):
    HiddenStates = GetStates(HEvents,N)
    P = np.empty((HiddenStates.shape[0]), dtype=np.float64)
    for i in range(P.shape[0]):
        P[i] = GetProb(T,E,Obs,HiddenStates[i],Prior)
    
    S= np.arange(0,HiddenStates.shape[0])

    M = np.where( P == np.amax(P))[0][0]
    SObs = GetS(Obs, DictO)
    if graph:
        plt.plot(S,P, color='black')
        plt.title(f'Probabilidad de cada secuencia \noculta para el estado observado \n{SObs}. \nPrior={Prior}')
        plt.axhline(P[M], color='red')
        plt.xlabel(f'Secuencia oculta')
        plt.ylabel(f'P')
        plt.grid()
        plt.show()
    return HiddenStates, P, M
    
def GetObsStatesInfo(OEvents,HEvents,N, T, E, Prior, graph=True):
    HiddenStates = GetStates(HEvents,N)
    
    ObsStates = GetStates(OEvents,N)

    Nobs = ObsStates.shape[0]
    PObs = np.zeros(Nobs)

    for j in range(Nobs):
        
        dim = HiddenStates.shape[0]
        P = np.zeros(dim)
        
        for i in range(dim):
            P[i] = GetProb(T,E,ObsStates[j],HiddenStates[i],Prior)
            
        PObs[j] = np.sum(P)

    S= np.arange(0,Nobs)

    M = np.where( PObs == np.amax(PObs))[0][0]
    
    if graph:
        plt.plot(S,PObs, color='black')
        plt.title(f'Probabilidad total de cada secuencia observable.\nPrior={Prior}')
        plt.axhline(PObs[M], color='red')
        plt.xlabel(f'Secuencia observable')
        plt.ylabel(f'P')
        plt.grid()
        plt.show()
    
    return ObsStates, PObs, M

T = np.array([[0.8,0.2],[0.2,0.8]])
E = np.array([[0.5,0.9],[0.5,0.1]])

C=0
S=1
J=0
B=1
OStates = np.array([C,S])
HStates = np.array([J,B])

#a
Prior = np.array([0.2,0.8])

DictH = {0:'Justa',1:'Sesgada'} 
DictO = {0:'Cara', 1:'Sello'}

Obs = np.array([S,C,C,C,S,C,S,C])

#b
HiddenStates, P, M = GetHiddenStatesInfo(HStates, Obs.size, Obs, T,E,Prior, DictO)
print(f'b: La secuencia oculta de monedas más probable para el estado observado {GetS(Obs,DictO)} es: \n{GetS(HiddenStates[M],DictH)} y su probabilidad es de {np.round(P[M],8)}')

#c
N=8
ObsStates, PObs, M = GetObsStatesInfo(OStates, HStates, N, T, E, Prior)
print(f'c: Las probabilidades se encuentran en la gráfica. \nLa secuencia observable de {N} eventos más probable es {GetS(ObsStates[M],DictH)}')

#d
print(f'La suma de las probabilidades de cada secuencia observable es {np.round(np.sum(PObs),3)}')

#e
HiddenStates, P, M = GetHiddenStatesInfo(HStates, Obs.size, Obs, T,E,Prior, DictO,False)
Prior2 = np.array([0.5,0.5])
HiddenStates2, P2, M2 = GetHiddenStatesInfo(HStates, Obs.size, Obs, T,E,Prior2, DictO,False)
print(f'Suponga el estado observado: {GetS(HiddenStates[M],DictH)}')
print(f'Vea que para un prior de {Prior}, la secuencia oculta de monedas más probable es: \n{GetS(HiddenStates[M],DictH)} y su probabilidad es de {np.round(P[M],8)}')
print(f'Pero para un prior distinto, por ejemplo, prior={Prior2}, la secuencia oculta de monedas más probable es: \n{GetS(HiddenStates2[M2],DictH)} y su probabilidad es de {np.round(P2[M2],8)}')
print(f'Luego los resultados si dependen de la probabilidad a priori.')


