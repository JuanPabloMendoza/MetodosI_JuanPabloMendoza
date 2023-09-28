import sympy as sym

x = sym.Symbol('x',real=True)
h = sym.Symbol('h',real=True)
y = sym.Symbol('f^(4)/e',real=True)

print(f'Se puede realizar la sustitucion u=x-x_0, donde ahora para x=x_0, u=0 y para x=x_3, u=3*h,')
print(f'para transformar la funcion integral de (x-x_0)(x-x_1)(x-x_2)(x-x_3) dx desde x_0 hasta x_3 en')
print(f'la integral de x(x-h)(x-2h)(x-3h) dx desde 0 hasta 3h')
f = x*(x-h)*(x-2*h)*(x-3*h)
Int = sym.integrate(f, (x, 0, 3*h))
print(f'El error local está dado por {(Int/24)*y}')