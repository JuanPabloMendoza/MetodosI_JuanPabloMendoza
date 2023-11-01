import numpy as np
import matplotlib.pyplot as plt

def GetExperiments(N=1000):
    
    freq = np.zeros(N)
    
    for i in range(int(N)):
        
        d1 = np.random.randint(1,7)
        d2 = np.random.randint(1,7)
        
        freq[i] = d1+d2
    
    return freq

freq = GetExperiments()

x = np.linspace(2,13,12)
h,bins = np.histogram(freq,bins=x)
w = np.diff(bins)
plt.bar(bins[:-1],h,width=w,ec='k')
plt.show()
print(bins)
print(h/1000.)


fr = 0
for i in range(len(bins)):
    if bins[i]<=7:
        fr+=h[i]
print(fr/1000.)
