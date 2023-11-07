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

def Minimizer(f,seed, N = 100, gamma = 0.001, momentum=0.695):
    
    r = np.zeros((N,2))
    r[0] = seed
    
    Grad = np.zeros((N,2))
    Grad[0] = Gradient(f,r[0,0],r[0,1])
    
    # We save the gradient in each step

    for i in tqdm(range(1,N)):
        r[i] = r[i-1] - gamma*Gradient(f,r[i-1,0],r[i-1,1]) + momentum*(r[i-1]-r[i-2])
        Grad[i] = Gradient(f,r[i-1,0],r[i-1,1])
        
        
    return r,Grad

N = 150
seed = np.array([0,3],dtype='float64')
r,Grad = Minimizer(f2,seed,N)

print(r)

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
    print(i)
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
filename='/Users/juanpablomendozaarias/Library/Mobile Documents/com~apple~CloudDocs/Universidad/Tercer Semestre/Métodos Computacionales/MetodosI_JuanPabloMendoza/Parcial 3/Aprendizaje Automático/Gradient2DMomentum.mp4'
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800 )
Animation.save( filename, writer=writer )

print(f'Revisando la información del gradiente, con momentum de 0.65, el algoritmo parece converger lo más rápido posible. Podría decirse que converge más del doble de rápido que sin momentum.')
print(f'Con el algoritmo original, tarda casi 400 iteraciones en alcanzar el mínimo')
print(f'Con un momentum de 0.6, tarda aproximadamente 145 iteraciones.')
print(f'Con un momentum de 0.65, tarda aproximadamente 135 iteraciones.')
print(f'Con un momentum de 0.68, tarda aproximadamente 130 iteraciones.')
print(f'Con los siguientes, el algoritmo converge a otra solución.')
print(f'Con un momentum de 0.8, tarda aproximadamente 115 iteraciones.')
print(f'Con un momentum de 0.82, tarda aproximadamente 110 iteraciones.')