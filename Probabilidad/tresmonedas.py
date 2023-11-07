import numpy as np
import matplotlib.pyplot as plt

def GetSample(N = int(1e4), ncoins = 1000, Weights = None):
    
    Sample = np.zeros((N,ncoins))
    
    Events = [0,1]
    
    for i in range(N):
        
        if Weights == None:
            Exp = np.random.choice(Events, ncoins)
        
        Sample[i] = Exp
    return Sample

Sample = GetSample()

Frecuencias = np.array([],dtype=np.int64)

for i in range(Sample.shape[0]):
    
    NCaras = 0
    
    for j in range(Sample.shape[1]):
        
        if Sample[i,j] == 0:
            NCaras += 1
            
    Frecuencias = np.append(Frecuencias,NCaras)

ii = np.where( Frecuencias == 2 )
Events = len(Frecuencias[ii])
print(Events/10000)
plt.hist(Frecuencias,density=True)
plt.show()