import numpy as np
import matplotlib.pyplot as plt
import time

Events = list(range(1,53))
print(Events)

def GetSample(N = int(1e7), ncartas = 5, Weights = None):
    Sample = np.zeros((N,ncartas))
    
    for i in range(N):
        
        if Weights == None:
            Exp = np.random.choice(Events, ncartas, replace=False)
        
        Sample[i] = Exp
    return Sample


N=int(1e6)
Sample = GetSample(N=N)

Frecuencias = np.array([],dtype=np.int64)
frecuencia=0

for i in range(Sample.shape[0]):
    nases = 0
    nreyes = 0
    for j in range(Sample.shape[1]):
        if Sample[i,j]%13 == 1:
            nases +=1
        elif Sample[i,j]%13 == 0:
            nreyes += 1
    if nases==3 and nreyes==2:
        frecuencia+=1

print(frecuencia/N)

""" for i in range(Sample.shape[0]):
    
    NCaras = 0
    
    for j in range(Sample.shape[1]):
        
        if Sample[i,j] == 0:
            NCaras += 1
            
    Frecuencias = np.append(Frecuencias,NCaras) """

""" ii = np.where( Frecuencias == 2 )
Events = len(Frecuencias[ii])
print(Events/10000)
plt.hist(Frecuencias,density=True)
plt.show() """