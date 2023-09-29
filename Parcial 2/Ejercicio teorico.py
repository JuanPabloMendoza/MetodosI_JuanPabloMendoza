import numpy as np
import sympy as sym

x = sym.Symbol('x',real=True)
#a
n=2
Polinomio = (sym.exp(x)/n)*sym.diff(sym.exp(-x)*x**n, x, 2)
print(Polinomio)

#b
raices = sym.solve(Polinomio, x)
print(raices)

#c
peso_1 = sym.integrate(sym.exp(-x)*(x-raices[1])/(raices[0]-raices[1]), (x,0,sym.oo))
peso_2 = sym.integrate(sym.exp(-x)*(x-raices[0])/(raices[0]-raices[1]), (x,0,sym.oo))
print(peso_1, peso_2)

#d
#La integral se puede realizar por metodo de integracion por partes. La funcion gamma de 4 (que corresponde a x elevado a la 3) da como resultado 3!=6
f = x**3
#por Gauss Laguerre:
Int = peso_1*raices[0]**3 + peso_2*raices[1]**3
print(Int)
print(f'Resolviendo esta expresion, encontramos que es 6')