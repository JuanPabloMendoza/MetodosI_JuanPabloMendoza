import numpy as np
import matplotlib.pyplot as plt
Prior = np.array([0.4, 0.3, 0.2, 0.1])
Lambda = np.array([1.,2.,3.,4.])

hatp = np.sum(Prior*Lambda)


poisson = lambda l,x: np.exp(-l) * l**x /np.math.factorial(x)

Likelihood = poisson(Lambda, 4)
Likelihood = poisson(Lambda,5)*Likelihood
Probx = np.sum(Likelihood*Prior)
Posterior = Likelihood*Prior/Probx
lhat = np.sum(Posterior*Lambda)
plt.stem(Posterior)
plt.show()