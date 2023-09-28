import numpy as np

def Integral_GL_corta(funcion, n):
    # Más rápida. Usa los pesos y raices dados por numpy
    raices,pesos = np.polynomial.legendre.leggauss(n)
    suma = 0
    for i in range(n):
        suma += pesos[i] * funcion(raices[i])
    return suma

Td= 300
kb= np.float128(1.3806488e-23)



def ecuacion_banda_prohibida():
    dT=1e-5
    N0V = 0.3
    Tc=11
    while Tc<20:
        def fun_int(x, banda=0.000001, T=Tc):
            banda_prima = Td*kb*banda
            return np.tanh((np.sqrt((x**2)+banda_prima**2)*Td)/(2*T))/np.sqrt((x**2)+(banda_prima**2))
        I = 0.5*Integral_GL_corta(fun_int, 50)
        print(np.abs(I-1/(N0V)))
        if np.abs(I-1/(N0V)) < dT:
            return Tc
        Tc+= dT
    return Tc
print(ecuacion_banda_prohibida())
    
    
