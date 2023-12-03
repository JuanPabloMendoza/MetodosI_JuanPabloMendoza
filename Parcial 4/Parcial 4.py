import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import copy
from tqdm import tqdm


#solucion del profesor
""" w1=[[-7.91098246,8.1744492, 0.08565776, 6.58849953, -5.2372043]]
b1=[[-4.6821248,-8.5133667,5.96705606 ,6.3788962, 3.63560092]]
w2=[[9.17252713],[9.8291565],[-0.64826691],[-4.94580118],[-5.44159593]]
b2=[[4.18317271]]
 """

#-----
#Sorting algorithm, quicksort.
#Sort by fitness
#codigo de https://stackabuse.com/quicksort-in-python/
def partition(array, start, end):
    pivot = array[start].Fitness
    low = start + 1
    high = end

    while True:
        # If the current value we're looking at is larger than the pivot
        # it's in the right place (right side of pivot) and we can move left,
        # to the next element.
        # We also need to make sure we haven't surpassed the low pointer, since that
        # indicates we have already moved all the elements to their correct side of the pivot
        while low <= high and array[high].Fitness >= pivot:
            high = high - 1

        # Opposite process of the one above
        while low <= high and array[low].Fitness <= pivot:
            low = low + 1

        # We either found a value for both high and low that is out of order
        # or low is higher than high, in which case we exit the loop
        if low <= high:
            array[low].Fitness, array[high].Fitness = array[high].Fitness, array[low].Fitness
            # The loop continues
        else:
            # We exit out of the loop
            break

    array[start].Fitness, array[high].Fitness = array[high].Fitness, array[start].Fitness

    return high

def quick_sort(array, start, end):
    if start >= end:
        return

    p = partition(array, start, end)
    quick_sort(array, start, p-1)
    quick_sort(array, p+1, end)
    
    
    
sigm = lambda x: 1/(1+np.exp(-x))

def FitnessFun(self):
    if self.Steps>0:
        return 1/(self.Steps+np.linalg.norm(self.r))
    else:
        return np.inf
 


def Punishment():
    pass

class Layer:
    
    def __init__(self,NC,NN,ActFun,rate=0.01, param=None): # Jugar con la tasa de mutacion
        
        self.NC = NC
        self.NN = NN
        self.ActFunc = ActFun
        self.rate = rate
        
        if param==None:
            self.W = np.random.uniform( -10.,10.,(self.NC,self.NN) )
            self.b = np.random.uniform( -10.,10.,(1,self.NN) )
        else:
            self.W = param[0]
            self.b = param[1]
        

        
    def Activation(self,x):
        z = np.dot(x,self.W) + self.b
        return self.ActFunc( z )[0]
    
    def Mutate(self):
    
        #self.W += np.random.normal( loc=0., scale=self.rate, size=(self.NC,self.NN))
        #self.b += np.random.normal( loc=0., scale=self.rate, size=(1,self.NN))
        
        """ self.W += np.random.uniform( -self.rate, self.rate, size=(self.NC,self.NN))
        self.b += np.random.uniform( -self.rate, self.rate, size=(1,self.NN)) """
        
def GetBrain(param=None):
    if not param==None:
        l0 = Layer(1,5,sigm,param=param[0])
        l1 = Layer(5,1,sigm,param=param[1])
    else:
        l0 = Layer(1,5,sigm)
        l1 = Layer(5,1,sigm)
    #l2 = Layer(2,1,sigm)
    Brain = [l0,l1]
    return Brain  

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
        
        #Numero de activaciones (para poder penalizar)
        self.NAct = 0
        
        self.Best = False
        
        
    def GetR(self):
        return self.r
    
    def Evolution(self):
        self.r += self.v*self.dt # Euler integration (Metodos 2)
        #-------
        """ if not np.linalg.norm(self.v)==0:
            self.Steps+=1 """
        #-------
        
        # Cada generación regreamos el robot al origin
        # Y volvemos a estimar su fitness
    def Reset(self, best=False):
        self.Steps = 0.
        self.r = np.array([0.,0.])
        self.NAct = 0
        theta = 0.
        self.v = np.array([1.*np.cos(theta),1.*np.sin(theta)])
            
    # Aca debes definir que es mejorar en tu proceso evolutivo
    def SetFitness(self):
        self.Fitness = FitnessFun(self)

        
       # Brain stuff
    def BrainActivation(self,x, downthreshold=0.7): 
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
        if self.Activation[0] > downthreshold:
            self.v = -self.v
            self.NAct +=1
            self.Steps+=0.1
        else:
            self.Steps +=1
        """    
        if self.NAct >=8:
            self.Steps-=0.5 """
        
            # Deberias penalizar de alguna forma, dado que mucha activación es desgastante!
            # Para cualquier cerebro
    
        return self.Activation
    
    # Aca mutamos (cambiar de parametros) para poder "aprender"
    def Mutate(self):
        for i in range(len(self.Layers)):
            self.Layers[i].Mutate()
    
    # Devolvemos la red neuronal ya entrenada
    def GetBrain(self):
        return self.Layers
    
def GetRobots(N,param=None):
    
    Robots = []
    
    for i in range(N):
        
        Brain = GetBrain(param=param)
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

def TimeEvolution(Robots,e,Plot=True):
    
    for it in range(t.shape[0]):
        
        if Plot:
        
            clear_output(wait=True)
        
            ax,ax1 = GetPlot()
            ax1.set_ylim(0.,1.)
        
            ax.set_title('t = {:.3f}'.format(t[it]))
        
        Activation = np.zeros(len(Robots))
        
        for i,p in enumerate(Robots):
            p.Evolution()
            if p.r[0]<1 and p.r>-1:
                p.Steps+=1
                
            # Activacion cerebral
            Act = p.BrainActivation(p.GetR()[0])
            Activation[i] = Act
            # Region donde aumentamos los pasos para el fitness
            
            
            if Plot and i < 10: # Solo pintamos los primeros 5, por tiempo de computo
                ax.scatter(p.r[0],p.r[1],label='Id: {}, Steps: {:.0f}, F: {:.0f}'.format(p.Id,p.Steps, p.Fitness))
                ax.quiver(p.r[0],p.r[1],p.v[0],p.v[1])
                
        # Pintamos la activaciones de los primeros 5
        
        if Plot:
            ax1.plot(np.arange(0,len(Robots[:5]),1),Activation[:5],marker='o',color='b',label='Activation')
            ax1.axhline(y=0.7,color='r')
        
        if Plot:
        
            ax.legend(loc=0)  
            ax1.legend(loc=0)
            plt.ion()
            plt.show()
            plt.pause(.01)
            plt.close()
        

# Definimos la rutina de entrenamiento
def Genetic(Robots, epochs = 200, Plot = True, Plottime=False):
    
    # Porcentaje de robots elegidos en cada epoch
    N = int(0.7*len(Robots))
    
    FitVector = np.array([])
    
    
    x = np.linspace(-1,1,20)
    Act = np.zeros_like(x)
    
    for e in range(int(epochs)):
        # Reiniciamos y mutamos los pesos
        
        
        for p in Robots:
            #----
            p.Reset() 
            p.Mutate()

            
            
        # Evolucionamos
        TimeEvolution(Robots,e,Plottime) # Apagar dibujar la evolución para entrenar
        
        # Actualizamos fitness de cada robot
        for i,p in enumerate(Robots):
            p.SetFitness()
        
        Robots_ = copy.deepcopy(Robots)
        Robots_[0].Best=False
        # Aca va toda la rutina de ordenar los bots del más apto al menos apto
        quick_sort(Robots_, start=0, end=len(Robots) - 1)
        Robots = Robots_
        # Guardamos el mejor fitness y el mejor robot

        best_fitness = Robots[0].Fitness

        best_bot = Robots[0] 
        best_bot.Best = True
        
        FitVector = np.append(FitVector,best_fitness)
        
        for i in range(len(x)):
            Act[i] = best_bot.BrainActivation(x[i])
        
        clear_output(wait=True)
        
        #print('Epoch:', e)
                
        # Last fitness
        print('Last Fitness:', FitVector[-1])
        print('Steps: ', best_bot.Steps)
        print('NAct: ', best_bot.NAct)
        
        
        if Plot:
            
            ax,ax1 = GetPlot()
            ax.plot(x,Act,color='k')
            ax.set_ylim(0.,1)
            ax.axhline(y=0.75,ls='--',color='r',label='Threshold')
            
            ax1.set_title('Fitness')
            ax1.plot(FitVector)
        
            ax.legend(loc=0)
            plt.ion()
            plt.show()
            plt.pause(.01)
            plt.close()
            
    
    
    return best_bot, FitVector

dt = 0.1
t = np.arange(0.,1.,dt)
Robots = GetRobots(10)
Best, FitVector = Genetic(Robots,Plot=False,Plottime=False) # Apagar Plottime para el entrenamiento


""" possible_params=[]
encontrar=False
if encontrar:
    for i in range(int(2e2)):
        print(i)
        Robots = GetRobots(10)
        Best, FitVector = Genetic(Robots,Plot=False,Plottime=False) 

        if Best.NAct<=35 and Best.NAct>10 and Best.Steps>15 and Best.Steps<40 :
            print(Best.Layers[0].W)
            print(Best.Layers[0].b)
            print(Best.Layers[1].W)
            print(Best.Layers[1].b)
            print(Best.NAct)
            print(Best.Steps)
            print(f'.')
            param=[[Best.Layers[0].W,Best.Layers[0].b],[Best.Layers[1].W,Best.Layers[1].b]]
            possible_params.append(param)
            #Robots = GetRobots(10, param=param)
            #Best, FitVector = Genetic(Robots,Plot=False,Plottime=True) 

print(possible_params)

#opcion 1
w1=[[-2.27436679,  1.09562916,  8.05783531,  9.39200248,  7.93081062]]
b1=[[-9.6033567,   9.13448236,  4.18667455,  0.36085966, -5.85240289]]
w2=[[ 6.93343131],
 [ 2.50471295],
 [ 1.56473449],
 [-7.64729648],
 [ 3.99522076]],
b2=[[2.08410865]] 
param1=[[w1,b1],[w2,b2]]


#opcion 2
w1=[[ 8.50828129,  6.91784772,  7.53230176, -1.80617305,  0.02935428]]
b1=[[ 0.76977676, -4.33576789, -0.90972329, -2.78746058,  3.93894297]]
w2=[[-1.11528753],
       [ 9.06591425],
       [ 3.90228796],
       [ 6.11051162],
       [-9.45278442]]
b2=[[8.86137451]]
param2=[[w1,b1],[w2,b2]]

#opcion 3

Robots = GetRobots(1, param2)
Best, FitVector = Genetic(Robots,Plot=True,Plottime=True) """
