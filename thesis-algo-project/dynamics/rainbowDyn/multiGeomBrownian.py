import numpy as np

# multi-asset Geometric Brownian motion
class multiGeometric:

    def __init__(self, s0=None, T=None, N=None, cov=None, dt=None):
        self.s0 = s0
        self.T = T
        self.N = N
        self.cov = cov
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

            processes[i, :] = processes[i - 1, :] - drift + diffusion

        return np.exp(processes)

    # function that loops through the afordefined to generate multiple paths
    def gen_path(self, nbPaths=None):

        dualPaths = np.zeros(((nbPaths, self.N + 1, 2)))

        for i in range(nbPaths):
            dualPaths[i, :, :] = self.gen_process()

        return dualPaths

# should be good, not tested yet