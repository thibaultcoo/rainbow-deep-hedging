import numpy as np

# multi-asset jump diffusion model
class multiJumpDiffusion:

    def __init__(self, m=None, s0=None, T=None,N=None,
                 cov=None, prob1=None, prob2=None, dt=None):

        self.m = m
        self.s0 = s0
        self.T = T
        self.N = N
        self.cov = cov
        self.prob1 = prob1
        self.prob2 = prob2
        self.dt = dt

    # function that generates a specific process
    def gen_process(self):

        sigma = np.zeros(2)
        drift = np.zeros(2)
        diffusion = np.zeros(2)

        processes = np.zeros((self.N + 1, 2))
        processes[0, :] = np.log(self.s0)

        for i in range(1, self.N + 1):
            z1 = np.random.normal(0, 1)
            sigma[0] = np.sqrt(cov[0, 0])
            sigma[1] = np.sqrt(cov[1, 1])

            drift = 0.5 * (sigma ** 2) * dt

            diffusion = np.sqrt(self.dt) * \
                        np.matmul(np.linalg.cholesky(self.cov),
                        np.random.normal(0, 1, size=2))

            jump = self.gen_jump(sigma[0], sigma[1])

            processes[i, :] = processes[i - 1, :] - drift + diffusion + jump

        return np.exp(processes)

    # function that generates a specific jump process
    def gen_jump(self, sigma1=None, sigma2=None):

        jump = np.zeros(2)
        zero = np.zeros(2)
        z = np.zeros(3)

        z[0] = 1
        z[1] = 2
        z[2] = 3

        val = np.random.choice(z, 1, p=[1 - self.prob1 - self.prob2,
                               self.prob1, self.prob2])

        if val == 1:
            jump = (self.m * np.random.exponential() +
                    np.sqrt(np.random.exponential()) *
                    np.random.multivariate_normal(zero, self.cov))
        elif val == 2:
            jump = [
                sigma1 * np.random.exponential() +
                np.sqrt(np.random.exponential()) *
                np.random.normal(0, sigma1 ** 2),0]
        elif val == 3:
            jump = [0, sigma2 * np.random.exponential() +
                    np.sqrt(np.random.exponential()) *
                    np.random.normal(0,sigma2 ** 2)]

        return np.random.poisson(1 * self.dt, size=2) * jump

    # function that loops through the afordefined to generate multiple paths
    def gen_path(self, nbPaths=None):

        dualPaths = np.zeros(((nbPaths, self.N + 1, 2)))

        for i in range(nbPaths):
            dualPaths[i, :, :] = self.gen_process()

        return dualPaths

# should be good, not tested yet