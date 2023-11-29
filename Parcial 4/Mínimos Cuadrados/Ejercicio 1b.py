import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


f1 = lambda x: 2*x-2
f2 = lambda x: 0.5-0.5*x
f3 = lambda x: 4-x

f1_ = lambda x,y: 2*x-y-2
f2_ = lambda x,y: x+2*y-1
f3_ = lambda x,y: x+y-4

D = lambda x,y: np.sqrt(f1_(x,y)**2 +f2_(x,y)**2 +f3_(x,y)**2 )

h = 0.01
X = np.linspace(-5.,5.,int(1e3))
Y = np.linspace(-5.,5.,int(1e3))


X_, Y_ = np.meshgrid(X, Y)
Z = D(X_,Y_)
ind = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
MinX = np.round(X[ind[1]],4)
MinY = np.round(Y[ind[0]],4)
MinDist = np.round(Z[ind],4)

F1 = f1(X)
F2 = f2(X)
F3 = f3(X)
plt.plot(X,F1,label='y=2x-2')
plt.plot(X,F2,label='y=0.5-0.5x')
plt.plot(X,F3,label='y=4-x')
plt.legend()
plt.scatter(MinX,MinY, label = f'Distancia = {MinDist}', color='red')
plt.legend()
plt.grid()
plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X_, Y_, Z, cmap=cm.coolwarm)
ax.scatter(MinX,MinY,MinDist,color ='black', label = f'Min: (X={MinX}, Y={MinY}), Dist={MinDist}')
ax.legend()
plt.show()

print(f'Punto encontrado: [{MinX},{MinY}]. Distancia = {MinDist}')
