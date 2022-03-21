#!/usr/bin/env python
# coding: utf-8

# In[2]:


# in the following we will try to transform the QuantLib BlackScholesProcess
# into a simple Numpy BlackScholesProcess
# from then on, we will be able to generate the other paths we will be interested in

# Heston dynamics
# Multi-asset geometric Brownian motion dynamics
# Kou multi-asset jump-diffusion model

import numpy as np

# Geometric Brownian Motion
class GeometricBrownianMotion:
    
    def __init__(self, s0 = None, sigma = None, r = None, matu = None, N = None):
        self.s0 = s0
        self.sigma = sigma
        self.r = r
        self.matu = matu
        self.N = N
        
    # function that generates a specific process
    def gen_process(self):
        
        dt = self.matu/self.N
        process = np.zeros(self.N+1)
        process[0] = np.log(self.s0)
        
        for i in range(1,self.N+1):
            z1 = np.random.normal(0,1)
            process[i] = process[i-1]-0.5*(self.sigma**2)*dt+self.sigma*np.sqrt(dt)*z1
        
        return np.exp(process)
    
    # function that loops through the afordefined to generate multiple paths
    def gen_path(self, nbPaths = None):
        
        paths = np.zeros((nbPaths, self.N+1))
        
        for i in range(nbPaths):
            paths[i,:] = self.gen_process()
        
        return paths

s0 = 100
sigma = 0.2
r = 0.0
matu = 1
N = 30
nbPaths = 3

geo = GeometricBrownianMotion(s0 = s0, sigma = sigma, r = r, matu = matu, N = N)
sto = geo.gen_path(nbPaths = nbPaths)
# working fine