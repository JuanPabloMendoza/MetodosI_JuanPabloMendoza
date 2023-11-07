import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from tqdm import tqdm

def f2(x,y):
    return x**4 + y**4 - 2*(x-y)**2
    #return 14*x**2 - 2*x**3 + 2*y**2 + 4*x*y
    
Dx = lambda f,x,y,h=1e-5: (f(x+h,y) - f(x-h,y))/(2*h)
Dy = lambda f,x,y,h=1e-5: (f(x,y+h) - f(x,y-h))/(2*h)


x0, y0 = 0.5,0.1

Gradient = lambda f,x,y: np.array([Dx(f,x,y),Dy(f,x,y)])
Gradient(f2,x0,y0)

def Minimizer(f,seed, N = 100, gamma = 0.01):
    
    r = np.zeros((N,2))
    r[0] = seed
    
    Grad = np.zeros((N,2))
    Grad[0] = Gradient(f,r[0,0],r[0,1])
    
    # We save the gradient in each step

    for i in tqdm(range(1,N)):
        r[i] = r[i-1] - gamma*Gradient(f,r[i-1,0],r[i-1,1])
        Grad[i] = Gradient(f,r[i-1,0],r[i-1,1])
        
        
    return r,Grad

N = 200
seed =np.array([0.,3.],dtype='float64')
r,Grad = Minimizer(f2,seed,N)

fig = plt.figure(figsize=(8,3))
ax = fig.add_subplot(1,2,1, projection = '3d',elev = 50, azim = -70)
ax1 = fig.add_subplot(1,2,2)

x = np.linspace(-3,3,100)
y = np.linspace(-3,3,100)
X,Y = np.meshgrid(x,y)
Z = f2(X,Y)

def init():
    ax.set_xlim3d(x[0],x[-1])
    ax.set_ylim3d(y[0],y[-1])
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')
    
def Update(i):
    
    ax.clear()
    ax1.clear()
    init()
    
    ax.set_title(r'$N=%.0f, Cost=%.3f$'%(i,f2(r[i,0],r[i,1])))
    ax.plot_surface(X,Y,Z, cmap = 'coolwarm', alpha=0.4)
    ax.scatter(r[:i,0],r[:i,1],f2(r[:i,0],r[:i,1]),marker='.',color='r')
    
    ax1.contour(X,Y,f2(X,Y))
    ax1.scatter(r[i,0],r[i,1],color='r') 
    ax1.quiver(r[i,0],r[i,1],-Grad[i,0],-Grad[i,1],color='r')


Animation = animation.FuncAnimation(fig, Update, frames=N,init_func=init)
filename='/Users/juanpablomendozaarias/Library/Mobile Documents/com~apple~CloudDocs/Universidad/Tercer Semestre/Métodos Computacionales/MetodosI_JuanPabloMendoza/Parcial 3/Aprendizaje Automático/Gradient2D.mp4'
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800 )
Animation.save( filename, writer=writer )