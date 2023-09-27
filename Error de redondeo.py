import numpy as np
import matplotlib.pyplot as plt
epsilon = 1
i=0
while 1+epsilon !=1:
    epsilon*=0.5
    i+=1
print(epsilon,i)

def epsilon(x):
    return np.sin(x) - (x-x**3/6.)
x=3*np.pi/4
print(epsilon(x)/epsilon(x/2))