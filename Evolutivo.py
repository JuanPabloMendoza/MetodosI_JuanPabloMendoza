import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

class Robot:
    def __init__(self, dt, Id=0) -> None:
        self.dt = dt
        self.Id = Id
        
        self.r = np.array([0., 0.])
        theta = np.random.uniform(0.2*np.pi)
        self.v = np.array([1*np.cos(theta), 1*np.sin(theta)])
        
    def Evolucion(self):
        self.r += self.v*self.dt
        
    def Evolucion_t(self, tiempo):
        ciclos = int(tiempo/self.dt)
        for i in range(ciclos):
            self.Evolucion()

        
        
dt = 0.05
t = np.arange(0., 2., dt)


def GetRobots(N):
    Robots = []
    for i in range(N):
        r = Robot(dt,Id=i)
        Robots.append(r)
    return Robots


Robots = GetRobots(50)
print(Robots)


def Plotter(t):
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    
    ax.set_title('t={:.2f}'.format(t))
    
    return ax


def TimeEvolution(Robots,t):
    for it in range(len(t)):
        
        clear_output(wait=True)
        
        ax = Plotter(t[it])
        
        
        for i, p in enumerate(Robots):
            
            p.Evolucion()
            
            ax.scatter(p.r[0],p.r[1],label='Id {}'.format(p.Id))
            ax.quiver(p.r[0],p.r[1],p.v[0],p.v[1])
    
        plt.show()
        
        time.sleep(0.01)
    ax.scatter

TimeEvolution(Robots,t)

