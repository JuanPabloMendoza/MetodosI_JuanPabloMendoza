import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

def f(x):
    return -x*((100-x)/2)

x = np.linspace(-10,70,200)
y = -f(x) 

plt.plot(x,y)

x0 = 5.
result = spo.minimize( f, x0)

plt.scatter(result.x[0], -result.fun)

plt.show()