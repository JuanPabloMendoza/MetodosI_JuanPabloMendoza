import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations
from itertools import combinations_with_replacement

# Definimos estados y priors
States = np.array([0,1]) # Deben ser enteros 
Prior =  np.array([0.4,0.6])

# Definimos matrices de emisión y transmision
T = np.array([[0.7,0.5],
              [0.3,0.5]])

E = np.array([[0.8,0.2],
              [0.1,0.3],
              [0.1,0.5]])

#np.sum(T,axis=0)

DictH = {0:'Feliz',1:'Triste'} 
DictH[0]


DictO = {0:'Rojo',1:'Verde',2:'Azul'}

Obs = np.array([1,2,0])

def GetStates(States,N):
    
    CStates = list( combinations_with_replacement(States,N) )
    
    print(CStates)
    
    Permu = []
    
    for it in CStates:
        p = list(permutations(it,N))
       # print(p)
        
        for i in p:
            if i not in Permu:
                Permu.append(i)
                
    return np.array(Permu)

ObsStates = GetStates(Obs, 3)
HiddenStates = GetStates(States,3)

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

P = np.zeros(HiddenStates.shape[0], dtype=np.float64)

for i in range(P.shape[0]):
    P[i] = GetProb(T,E,Obs,HiddenStates[i],Prior)
    
plt.plot(P)
plt.show()

ii = np.where( P == np.amax(P))
print(HiddenStates[ii])

ObsStates = GetStates([0,1,2],3)

Nobs = ObsStates.shape[0]

PObs = np.zeros(Nobs)

for j in range(Nobs):
    
    dim = HiddenStates.shape[0]
    P = np.zeros(dim)
    
    for i in range(dim):
        P[i] = GetProb(T,E,ObsStates[j],HiddenStates[i],Prior)
        
    PObs[j] = np.sum(P)

plt.plot(PObs)
plt.show()

ii = np.where( PObs == np.amax(PObs))
print(ObsStates[ii])