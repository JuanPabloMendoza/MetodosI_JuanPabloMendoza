import numpy as np

Eventos = np.array([-1,1])

def lanzamiento_de_monedas(nmonedas,N=int(1e5)):
    Sample = np.zeros((N,nmonedas))
    for i in range(N):
        for j in range(nmonedas):
            Sample[i,j] = np.random.choice(Eventos,1,replace=False)
    return Sample

nmonedas=4
Sample = lanzamiento_de_monedas(nmonedas)

N = Sample.shape[0]
fr = 0

C=2
S=nmonedas-C #S+C=nmonedas

for exp in Sample:
    ncaras = 0
    nsellos = 0
    for i in range(exp.size):
        if exp[i]==1:
            ncaras+=1
        else:
            nsellos+=1
    if ncaras==C and nsellos==S: #es inmediata la condición para nsellos
        fr+=1
        
Prob = fr/N
print(f'La probabilidad de obtener {C} caras y {S} sellos al lanzar {nmonedas} monedas es: {np.round(Prob*100,3)}%')
#con N=1e6 da casi exacto.