#!/usr/bin/env python
# coding: utf-8

# multi-asset jump diffusion model
import numpy as np
import matplotlib.pyplot as plt

class multiJumpDiffusion:

    def __init__(self, m=None, spot_init=None, T=None, N=None, cov=None, prob1=None, prob2=None):

        self.m = m
        self.spot_init = spot_init
        self.T = T
        self.N = N
        self.cov = cov
        self.prob1 = prob1
        self.prob2 = prob2

    def gen_process(self):

        dt = self.T / self.N
        sigma = np.zeros(2)
        processes = np.zeros((self.N, 2))
        processes[0, :] = np.log(self.spot_init)

        drift = np.zeros(2)
        diffusion = np.zeros(2)

        for i in range(1, self.N):
            z1 = np.random.normal(0, 1)
            sigma[0] = np.sqrt(cov[0, 0])
            sigma[1] = np.sqrt(cov[1, 1])

            drift = 0.5 * (sigma ** 2) * dt
            diffusion = np.sqrt(self.T / self.N) * np.matmul(np.linalg.cholesky(self.cov),
                                                             np.random.normal(0, 1, size=2))
            jump = self.gen_jump(sigma[0], sigma[1])

            processes[i, :] = processes[i - 1, :] - drift + diffusion + jump

        return np.exp(processes)

    def gen_jump(self, sigma1=None, sigma2=None):

        jump = np.zeros(2)
        a1 = np.zeros(2)
        a2 = np.zeros(2)
        a3 = np.zeros(2)
        zero = np.zeros(2)

        a1[:] = 0
        a2[:] = 0
        a3[:] = 0

        z = np.zeros(3)
        z[0] = 1
        z[1] = 2
        z[2] = 3

        val = np.random.choice(z, 1, p=[1 - self.prob1 - self.prob2, self.prob1, self.prob2])

        if val == 1:
            jump = (self.m * np.random.exponential() + np.sqrt(np.random.exponential()) * np.random.multivariate_normal(
                zero, self.cov))
        elif val == 2:
            jump = [
                sigma1 * np.random.exponential() + np.sqrt(np.random.exponential()) * np.random.normal(0, sigma1 ** 2),
                0]
        elif val == 3:
            jump = [0, sigma2 * np.random.exponential() + np.sqrt(np.random.exponential()) * np.random.normal(0,sigma2 ** 2)]

        return np.random.poisson(1 * self.T / self.N, size=2) * jump

    def gen_path(self, nbPaths=None):

        dt = np.divide(self.T, self.N)
        dualPaths = np.zeros(((nbPaths, self.N, 2)))

        for i in range(nbPaths):
            dualPaths[i, :, :] = self.gen_process()

        return dualPaths

m = np.array([0.1, 0.2])
spot_init = np.array([100, 100])
T = 1
N = 30
cov = np.array([[0.1, 0.05], [0.05, 0.1]])
prob1 = 0.2
prob2 = 0.2

test = multiJumpDiffusion(m=m, spot_init=spot_init, T=T, N=N, cov=cov, prob1=prob1, prob2=prob2)
hihi = test.gen_path(1)
# working fine
