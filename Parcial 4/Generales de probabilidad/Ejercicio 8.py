import numpy as np

Eventos = [-1,1]

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
for exp in Sample:
    ncaras = 0
    nsellos = 0
    for i in range(exp.size):
        if exp[i]==1:
            ncaras+=1
        else:
            nsellos+=1
    if ncaras==2 and nsellos==2: #sobra nsellos
        fr+=1
Prob = fr/N
print(f'La probabilidad de obtener 2 caras y 2 sellos al lanzar 4 monedas es: {Prob*100}%')
print(300/8)