import numpy as np
import sympy as sym

def Integral_Legendre(funcion,a,b,n):
    raices, pesos = np.polynomial.legendre.leggauss(n)
    raices = 0.5*(raices*(b-a)+(b+a))
    Integral = 0
    for i in range(n):
        #raiz_con_sustitucion = 0.5*(raices[i]*(b-a)+(b+a))
        Integral += pesos[i]*funcion(raices[i])
    return 0.5*(b-a)*Integral


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

def Triple_Integral_Legendre(funcion,a,b,c,d,e,f,n):
    raices, pesos = np.polynomial.legendre.leggauss(n)
    raices_x = 0.5*(raices*(b-a)+(b+a))
    raices_y = 0.5*(raices*(d-c)+(d+c))
    raices_z = 0.5*(raices*(f-e)+(f+e))
    Triple_Integral = 0
    for i in range(n):
        Doble_Integral_int = 0
        for j in range(n):
            Integral_int = 0
            for k in range(n):
                Integral_int += pesos[k]*funcion(raices_x[i], raices_y[j], raices_z[k])
            Doble_Integral_int += pesos[j]*Integral_int
        Triple_Integral += pesos[i]*Doble_Integral_int
    return 0.125*(b-a)*(d-c)*(f-e)*Triple_Integral
            
def funcion_1(x):
    return -x+2

def Densidad(x,y):
    return x+2*y**2

def funcion_3(x,y,z):
    return x*y*z

print(Integral_Legendre(funcion_1, 0, 2, 3))

print(Doble_Integral_Legendre(Densidad, a=1, b=3, c=1, d=4, n=7))  

print(Triple_Integral_Legendre(funcion_3, a=1, b=2, c=1, d=2, e=1, f=2, n=6))      

def Centro_de_masa(densidad, a, b, c, d, n):
    Masa = Doble_Integral_Legendre(densidad, a, b, c, d, n)
    print(Masa)
    def f_x(x,y):
        return densidad(x,y)*x
    def f_y(x,y):
        return densidad(x,y)*y
    Xm = (1/Masa) * Doble_Integral_Legendre(f_x, a, b, c, d, n)
    Ym = (1/Masa) * Doble_Integral_Legendre(f_y, a, b, c, d, n)
    
    return Xm, Ym

print(Centro_de_masa(Densidad, a=1, b=3, c=1, d=4, n=7))


def fun(x,y):
    return x

print(Doble_Integral_Legendre(fun, 0, 1, 0, 4, 15))







    
    