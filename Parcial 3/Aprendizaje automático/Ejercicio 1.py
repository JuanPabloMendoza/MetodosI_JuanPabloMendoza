import numpy as np
from sympy import *
init_printing(use_unicode=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = Symbol('x',real=True)
y = Symbol('y',real=True)


f = lambda x,y: x**2 - y**2 + 2*x


def Taylor_1(f, a, x, y):
    fx = lambdify([x,y],diff(f(x,y),x,1))
    fy = lambdify([x,y],diff(f(x,y),y,1))
    return lambda x,y: f(a[0], a[1]) +fx(a[0], a[1])*(x-a[0]) + fy(a[0], a[1])*(y-a[1])

a = [1,1]
PT = Taylor_1(f,a,x,y)



N=100
X = np.linspace(-20,20,N)
Y = np.linspace(-20,20,N)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(X, Y)
Z1 = PT(X,Y)
Z2 = f(X,Y)
ax.set_title(f'Curva z={f(x,y)}')
ax.plot_surface(X,Y,Z1, color = 'red', alpha=0.4)
ax.plot_surface(X,Y,Z2, color = 'blue', alpha=0.4)
ax.scatter(a[0],a[1],f(a[0],a[1]), color='black', marker='.', label=f'Plano tangente en {a}')
plt.legend()
plt.show()




