import numpy as np
import matplotlib.pyplot as plt

def Likelihood(x,mu,std):
    return 1/np.sqrt(2*np.pi*std**2)*np.exp(-(x-mu)**2/(2*std**2))

mu = 0.3
sigma = 1.5

t = np.linspace(-5,5,20)
Like = Likelihood(t,mu,sigma)


def Metropolis(Likelihood, mu_, sigma_, seed_=0., delta=1., N= int(1e5)):
    
    x = np.zeros(N)
    x[0] = seed_
    for i in range(1,N-1):
        
        # futuro - - - - - # present
        xn = x[i] + np.random.uniform(-delta, delta)
        
            #prob. aceptacion
        alpha = np.minimum(1 , Likelihood(xn,mu_,sigma_)/Likelihood(x[i],mu_,sigma_) )
        
        g = np.random.rand()
        if g<alpha:
            x[i+1] = xn
        else:
            x[i+1]= x[i]
            
    return x

x = Metropolis(Likelihood, mu, sigma)

plt.plot(t, Like)
plt.hist(x, density=True, bins=30)
plt.show()

    