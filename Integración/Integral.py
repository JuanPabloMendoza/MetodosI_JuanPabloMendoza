import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def Function(x):
    return np.sin(x)

def Integral(Funcion, a,b):
    return (b-a)*0.5*(Funcion(a)+Funcion(b))

print(Integral(Function, 0, np.pi/2))


class Integrator:
    def __init__(self, x, f):
        self.x = x
        self.h = self.x[1] - self.x[0]
        self.y = f(self.x)
        
        self.Integral = 0.
    
    def GetIntegral(self):
        self.Integral = 0.5*(self.y[0] + self.y[-1])
        
        self.Integral += np.sum( self.y[1:-1])
        self.Integral *= self.h
        
        return self.Integral
        


x = np.linspace(0, np.pi/2, 50)
a = 0
b = np.pi/2
integral = Integrator(x, Function)
print(integral.GetIntegral())

integrate.trapezoid(Function(x), x)
