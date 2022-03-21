#!/usr/bin/env python
# coding: utf-8

# In[10]:


# now we model the multi-asset Geometric Brownian motion for two assets
# generates a matrix: for each simulation we have two paths (for our two assets) of N steps
# covariance matrix needs to be positive defined (to get the Cholesky)

import numpy as np

# multi-asset Geometric Brownian motion
class multiGeometric:
    
    def __init__(self, s0 = None, T = None, N = None, cov = None):
        self.s0 = s0
        self.T = T
        self.N = N
        self.cov = cov
        
    # function that generates a specific process
    def gen_process(self):
        
        dt = np.divide(self.T,self.N)
        sigma = np.zeros(2)
        processes = np.zeros((self.N,2))
        processes[0,:] = np.log(self.s0)
        
        drift = np.zeros(1)
        diffusion = np.zeros(1)
        
        for i in range(1,self.N):
            z1 = np.random.normal(0,1)
            sigma[0] = cov[0,0]
            sigma[1] = cov[1,1]
            
            drift = 0.5*(sigma**2)*dt
            diffusion = np.sqrt(self.T/self.N)*np.matmul(np.linalg.cholesky(self.cov), np.random.normal(0,1,size=2))
            
            processes[i,:] = processes[i-1,:] - drift + diffusion
        
        return np.exp(processes)
    
    # function that loops through the afordefined to generate multiple paths
    def gen_path(self, nbPaths = None):
        
        dt = np.divide(self.T,self.N)
        dualPaths = np.zeros(((nbPaths,self.N,2)))
        
        for i in range(nbPaths):
            dualPaths[i,:,:] = self.gen_process()
        
        return dualPaths
    
#s0 = np.array([100,100])
#cov = np.array([[0.1, 0.05],[0.05, 0.2]])
    
#sto = multiGeometric(s0 = s0, T = 1, N = 30, cov = cov)
#spot = sto.gen_path(nbPaths = 3)
#path = sto.gen_process()
# working fine