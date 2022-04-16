import numpy as np

# single-asset Geometric Brownian motion
class GeometricBrownianMotion:

    def __init__(self, s0=None, sigma=None, r=None, matu=None, N=None, dt=None):
        self.s0 = s0
        self.sigma = sigma
        self.r = r
        self.matu = matu
        self.N = N
        self.dt = dt

    # function that generates a specific process
    def gen_process(self):

        process = np.zeros(self.N + 1)
        process[0] = np.log(self.s0)

        for i in range(1, self.N + 1):
            z1 = np.random.normal(0, 1)
            process[i] = process[i - 1] - 0.5 * (self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * z1

        return np.exp(process)

    # function that loops through the afordefined to generate multiple paths
    def gen_path(self, nbPaths=None):

        paths = np.zeros((nbPaths, self.N + 1))

        for i in range(nbPaths):
            paths[i, :] = self.gen_process()

        return paths