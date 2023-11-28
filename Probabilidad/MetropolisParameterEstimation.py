import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

def Likelihood(mu,n,b,s):
    
    l = mu*s + b
    
    L = np.exp( -l ) * l**n / np.math.factorial(int(n))
    
    return L


def LogPrior(p):
    mu = p
    
    if mu>=0.:
        return 0
    else:
        return -np.inf
def JointLogLikelihood(mu,data):
    
    N = data.shape[0]
    
    n = data[:,0]
    b = data[:,1]
    s = data[:,2]
    
    JointL = 0.
    
    for c in range(N):
        JointL += np.log(Likelihood(mu,n[c],b[c],s[c]))
        
    return JointL

def LogPosterior(p,data):
    
    LogP = LogPrior(p)
    
    if not np.isfinite(LogP):
        return -np.inf
    else:
        return JointLogLikelihood(p,data) + LogP
    
n = np.array([1,0,0])
b = np.array([0,0,0])
s = np.array([1,2,1])

N = n.shape[0]

data = np.zeros((N,3))

data[:,0] = n
data[:,1] = b
data[:,2] = s

mu = np.linspace(0.001,2.,1000)
Posterior = np.zeros_like(mu)

for i in range(len(mu)):
    Posterior[i] = LogPosterior(mu[i], data)
    
plt.plot(mu, Posterior)

nll = lambda *p : LogPosterior(*p)

nll(0.3, data)


n_walkers, n_params = 5,1
p0 = np.zeros((n_walkers, n_params))
p0[:,0]=1
p0 += np.random.rand(n_walkers, n_params)
sampler = emcee.EnsembleSampler(n_walkers, n_params, nll, args=[data])
pos, prob, state, _ = sampler.run_mcmc(p0, 10000, progress=True)


fig, axes = plt.subplots(n_params, figsize=(10, 5), sharex=True)


samples = sampler.get_chain()

labels = ["$p$"]


for i in range(n_params):
    #ax = axes[i]
    axes.plot(samples[:,:,i], "k", alpha=0.7)
    axes.set_xlim(0, len(samples))
    axes.set_ylabel(labels[i],rotation=0, fontsize=15)
    axes.yaxis.set_label_coords(-0.1, 0.5)

axes.set_xlabel("step number",fontsize=15)

flat_samples = sampler.get_chain( discard=500, flat=True )

truths = np.percentile(flat_samples, 50, axis=0)

figure = corner.corner(flat_samples,truths=truths, labels=labels, quantiles=[0.16,0.5,0.84], show_titles=True)

#plt.hist(samples[0])
print(samples)

