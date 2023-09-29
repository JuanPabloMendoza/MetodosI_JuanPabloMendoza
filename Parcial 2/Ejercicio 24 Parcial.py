import numpy as np
import matplotlib.pyplot as plt

k = 1.9429303960



def Doble_Integral_Legendre(funcion,a,b,c,d,n):
    raices, pesos = np.polynomial.legendre.leggauss(n)
    raices_x = 0.5*(raices*(b-a)+(b+a))
    raices_y = 0.5*(raices*(d-c)+(d+c))
    Doble_Integral = 0
    for i in range(n):
        Integral_int = 0
        for j in range(n):
            Integral_int += pesos[j]*funcion(raices_x[i], raices_y[j])
        Doble_Integral += pesos[i]*Integral_int
    return 0.25*(b-a)*(d-c)*Doble_Integral

def Fun(Posicion):
    x=Posicion[0]
    y=Posicion[1]
    z=Posicion[2]
    def Integrando(ang, r):
        return (z*r)/(((x**2)+(y**2)+(z**2)+(r**2)-2*r*x*np.cos(ang)-2*r*y*np.sin(ang))**1.5)
    return Integrando

N = 50
Posicion = np.array([0,0,0.2], dtype='float64')
print(-k*Doble_Integral_Legendre(Fun(Posicion), 0, 2*np.pi, 0, 1, 50))

R=np.array([0, 0.125, 0.25, 0.38, 0.5])
phi = np.linspace(0,2*np.pi, 10)
def comportamiento(R,phi):
    z=0.2
    for r in R:
        g = np.array([])
        for ang in phi:
            x=r*np.cos(ang)
            y=r*np.sin(ang)
            Posicion = [x,y,z]
            Int = -k*Doble_Integral_Legendre(Fun(Posicion), 0, 2*np.pi, 0, 1, 50)
            g = np.append(g, Int)
        plt.plot(phi, g, label = f'R = {r}')
        plt.legend()
    plt.show()

comportamiento(R,phi)
        
        