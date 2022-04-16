import numpy as np

# stochastic volatility Heston model
class Heston:

    def __init__(self, s0=None, v0=None, r=None, T=None, N=None, kappa=None,
                 theta=None, xi=None, rho=None, dt=None):
        self.s0 = s0
        self.v0 = v0
        self.r = r
        self.T = T
        self.N = N
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.dt = dt

    # function that generates a specific process
    def gen_process(self):

        spot_process = np.zeros(self.N + 1)
        vol_process = np.zeros(self.N + 1)

        spot_process[0] = np.log(self.s0)
        vol_process[0] = np.log(self.v0)

        for i in range(1, self.N + 1):
            zS = np.random.normal(0, 1)
            z1 = np.random.normal(0, 1)
            zV = self.rho * zS + np.sqrt(1 - self.rho ** 2) * z1

            spot_process[i] = spot_process[i - 1] + self.r - 0.5 * np.exp(vol_process[i - 1]) * self.dt + np.sqrt(
                              np.exp(vol_process[i - 1])) * np.sqrt(self.dt) * zS
            vol_process[i] = vol_process[i - 1] + np.divide(1, np.exp(vol_process[i - 1])) * \
                             self.kappa * (self.theta - np.exp(vol_process[i - 1])) - \
                             (0.5 * self.xi ** 2) * self.dt + \
                             np.divide(self.xi, np.sqrt(np.exp(vol_process[i - 1]))) * np.sqrt(self.dt) * zV

        return np.exp(spot_process), np.exp(vol_process)

    # function that loops through the afordefined to generate multiple paths
    def gen_path(self, nbPaths=None):

        dt = np.divide(self.T, self.N)
        spot_paths = np.zeros((nbPaths, self.N + 1))
        vol_paths = np.zeros((nbPaths, self.N + 1))

        for i in range(nbPaths):
            spot_paths[i, :], vol_paths[i, :] = self.gen_process()

        return spot_paths

sto = Heston(s0 = 100, v0 = 0.2, r = 0, T = 1, N = 30, kappa = 2, theta = 0.04, xi = 0.3, rho = 0.4, dt=1 / 365)
spot = sto.gen_path(nbPaths = 2)
path = sto.gen_process()
