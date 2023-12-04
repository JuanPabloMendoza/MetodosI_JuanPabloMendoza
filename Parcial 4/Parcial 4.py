import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import copy
from tqdm import tqdm

sigm = lambda x: 1/(1+np.exp(-x))

class Layer:
    
    
    def __init__(self,NC,NN,ActFun, param=None,rate=0.018): 
        
        self.NC = NC
        self.NN = NN
        self.ActFunc = ActFun
        self.rate = rate
        
        if param==None:
            self.W = np.random.uniform( -10.,10.,(self.NC,self.NN) )
            self.b = np.random.uniform( -10.,10.,(1,self.NN) )
        else:
            self.W=param[0]
            self.b=param[1]
        
    def Activation(self,x):
        z = np.dot(x,self.W) + self.b
        return self.ActFunc( z )[0]
    
    def Mutate(self):
        
        self.W += np.random.uniform( -self.rate, self.rate, size=(self.NC,self.NN))
        self.b += np.random.uniform( -self.rate, self.rate, size=(1,self.NN))
        
def GetBrain(param=None):
    if param==None:
        l0 = Layer(1,5,sigm)
        l1 = Layer(5,1,sigm)
    else:
        l0 = Layer(1,5,sigm, param[0])
        l1 = Layer(5,1,sigm, param[1])
    #l2 = Layer(2,1,sigm)
    Brain = [l0,l1]
    return Brain 

def Fitness(self):
    if self.NAct ==0:
        return np.inf
    elif -1.<=self.r[0]<=1:
        return 1/(self.Steps+np.abs(self.r[0]))
    else:
        return np.inf


class Robot:
    
    def __init__(self, dt, Layers, Id=0):
        
        self.Id = Id
        self.dt = dt
        
        
        self.r = np.random.uniform([0.,0.])
        theta = 0.
        self.v = np.array([1.*np.cos(theta),1.*np.sin(theta)])

        
        # Capacidad o aptitud del individuo
        self.Fitness = np.inf
        self.Steps = 0

        # Brain
        self.Layers = Layers
        
        self.NAct = 0
        
    def GetR(self):
        return self.r
    
    def Evolution(self):
        self.r += self.v*self.dt # Euler integration (Metodos 2)

        # Cada generación regreamos el robot al origin
        # Y volvemos a estimar su fitness
    def Reset(self):
        self.Steps = 0.
        self.r = np.array([0.,0.])
        self.Fitness = np.inf    
        
    # Aca debes definir que es mejorar en tu proceso evolutivo
    def SetFitness(self):
        self.Fitness = Fitness(self)
        
        
       # Brain stuff
    def BrainActivation(self,x,threshold): 
        # El umbral (threshold) cerebral es a tu gusto!
        # cercano a 1 es exigente
        # cercano a 0 es sindrome de down
        
        # Forward pass - la infomación fluye por el modelo hacia adelante
        for i in range(len(self.Layers)):         
            if i == 0:
                output = self.Layers[i].Activation(x)
            else:
                output = self.Layers[i].Activation(output)
        
        self.Activation = np.round(output,4)
    
        # Cambiamos el vector velocidad
        if self.Activation[0] > threshold:
            self.v = -self.v
            self.NAct+=1
            if -1.<=self.r[0]<=1.:
                self.Steps*=0.6
            elif self.r[0]>1. or self.r[0]<-1.:
                self.Steps *=0.05
            
    
        return self.Activation
    
    # Aca mutamos (cambiar de parametros) para poder "aprender"
    def Mutate(self):
        for i in range(len(self.Layers)):
            self.Layers[i].Mutate()
    
    # Devolvemos la red neuronal ya entrenada
    def GetBrain(self):
        return self.Layers
    
def GetRobots(N, param=None):
    
    Robots = []
    
    for i in range(N):
        
        Brain = GetBrain(param)
        r = Robot(dt,Brain,Id=i)
        Robots.append(r)
        
    return Robots

def GetPlot():
    
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,2,1)
    ax1 = fig.add_subplot(1,2,2)
    
    ax.set_xlim(-1.,1.)
    ax.set_ylim(-1.,1.)
 
    return ax,ax1

def TimeEvolution(Robots,e,threshold, Plot=True):
    
  
    for it in range(t.shape[0]):
        
        if Plot:
        
            clear_output(wait=True)
        
            ax,ax1 = GetPlot()
            ax1.set_ylim(0.,1.)
        
            ax.set_title('t = {:.3f}'.format(t[it]))
        
        Activation = np.zeros(len(Robots))
        
        for i,p in enumerate(Robots):
            p.Evolution()
            
            if -1.<=p.r[0]<=1.:
                p.Steps+=2
            # Activacion cerebral
            Act = p.BrainActivation(p.GetR()[0], threshold)
            Activation[i] = Act
            
                
            if Plot and i <= 10: 
                ax.scatter(p.r[0],p.r[1],label='Id: {}, Steps: {:.0f}'.format(p.Id,p.Steps))
                ax.quiver(p.r[0],p.r[1],p.v[0],p.v[1])
                
        
        if Plot:
            ax1.plot(np.arange(0,len(Robots[:5]),1),Activation[:5],marker='o',color='b',label='Activation')
            ax1.axhline(y=threshold,color='r')
        
        if Plot:
        
            ax.legend(loc=0)  
            ax1.legend(loc=0)
            
            plt.ion()
            plt.show()
            plt.pause(.001)
            plt.close()

def Genetic(Robots, epochs = 200, Plot = True, Plottime=False):
    
    # Porcentaje de robots elegidos en cada epoch
    N = int(0.7*len(Robots))
    
    FitVector = np.array([])
    
    threshold = 0.6
    x = np.linspace(-1,1,20)
    Act = np.zeros_like(x)
    
    for e in range(int(epochs)):
        
        for p in Robots:
            p.Reset() 
            p.Mutate()

        TimeEvolution(Robots,e,threshold,Plottime) # Apagar dibujar la evolución para entrenar

        for i,p in enumerate(Robots):
            p.SetFitness()


        scores = [ (p.Fitness,p) for p in Robots ]

        scores.sort(  key = lambda x: x[0], reverse = False  )
        
        Temp = [r[1] for i,r in enumerate(scores) if i < N]

        for i,r in enumerate(Robots):
            j = i%N
            Robots[i] = copy.deepcopy(Temp[j])
        

        best_fitness = Robots[0].Fitness
        best_bot = Robots[0] 
        
        FitVector = np.append(FitVector,best_fitness)
        
        for i in range(len(x)):
            Act[i] = best_bot.BrainActivation(x[i], threshold)
        
        clear_output(wait=True)
        
        #print('Epoch:', e)
                
        # Last fitness
        #print('Last Fitness:', FitVector[-1])
        
        
        if Plot:
            
            ax,ax1 = GetPlot()
            ax.plot(x,Act,color='k')
            ax.set_ylim(0.,1)
            ax.axhline(y=threshold,ls='--',color='r',label='Threshold')
            
            ax1.set_title('Fitness')
            ax1.plot(FitVector)
        
            ax.legend(loc=0)
            
            plt.ion()
            plt.show()
            plt.pause(.0001)
            plt.close()
        
        
    
    return best_bot, FitVector[-1]

dt = 0.1
""" t = np.arange(0.,5.,dt)
bestparam=None
fitness=np.inf
for i in range(20):
    print(i)
    Robots = GetRobots(80)
    Best, FitVector = Genetic(Robots, epochs = 1000, Plot=False,Plottime=False) 
    param=[[Best.Layers[0].W, Best.Layers[0].b],[Best.Layers[1].W, Best.Layers[1].b]]
    print(FitVector)
    if fitness>FitVector:
        fitness=FitVector
        bestparam=param
param=bestparam
print(fitness)
print(param)  """

#Este parece salirse un poco por el borde derecho. Si igualmente vale, está perfecto. Llega a cada borde y se devuelve.
param=[[[-3.643607  ,  2.50272155, -7.26860228, -6.27007524, -3.0053872 ],
        [-3.31453685,  2.98429363,  5.0174084 ,  6.52415002, -2.5937588 ]],
       [[[ 8.99713828],
         [ 8.01842524],
         [-0.80100321],
         [-9.707941  ],
         [ 5.79069946]],
        [[-2.59990745]]]]

#Este no llega justamente al borde, pero queda muy cerca.
param=[[[ 2.23838335,  9.68341207, -4.07036293, -1.92326365, 10.00468513],
        [-4.31106477, -9.25235598, -6.32161815,  7.14478022,  9.48326097]],
       [[[-6.15296162],
         [ 9.54750834],
         [ 6.01791882],
         [ 7.32028818],
         [-5.93729987]],
        [[-2.80715872]]]]


t = np.arange(0.,20.,dt)
Robots = GetRobots(2,param)
Best, FitVector = Genetic(Robots,epochs=1,Plot=False,Plottime=True)