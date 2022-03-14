import numpy as np
import scipy.stats as stats
from dynamics.multiJumpDiffusion import multiJumpDiffusion
from functions.bivariateNormal import bivnormcdf

class EuropeanRainbow:
    def __init__(self):
        pass

    # S -> spot prices of the two assets
    # cov -> constant variance-covariance matrix
    # r -> constant risk-free rate
    # K -> strike price of the rainbow option
    # matu -> maturity of the rainbow option
    # N -> number of steps between ignition and maturity
    # payoff -> final payoff of the rainbow option
    # eps -> proportional transaction cost (same for both assets)

    def get_Stulz_price(self, S=None, cov=None, r=None, K=None, matu=None, N=None, nbPaths=None):

        dt = matu / N
        T = np.arange(0, N) * dt
        T = np.repeat(np.flip(T[None, :]), S.shape[0], axis=0)

        spot1 = np.zeros((nbPaths, N))
        spot2 = np.zeros((nbPaths, N))

        eta1 = np.zeros((nbPaths, N))
        eta2 = np.zeros((nbPaths, N))

        beta1 = np.zeros((nbPaths, N))
        beta2 = np.zeros((nbPaths, N))

        gamma1 = np.zeros((nbPaths, N))
        gamma2 = np.zeros((nbPaths, N))

        spot1[:, :] = S[:, :, 0]
        spot2[:, :] = S[:, :, 1]

        sigma1 = np.sqrt(cov[0, 0])
        sigma2 = np.sqrt(cov[1, 1])
        rho = cov[0, 1] / (sigma1 * sigma2)

        sig = cov[0,0] + cov[1,1] - 2 * cov[0,1]
        rho_indiv1 = (rho * sigma2 - sigma1) / np.sqrt(sig)
        rho_indiv2 = (rho * sigma1 - sigma2) / np.sqrt(sig)

        price = np.zeros((nbPaths, N-1))

        with np.errstate(divide="ignore"):
            eta1 = np.divide(np.log(spot1 / K) + (r + 0.5 * (sigma1 ** 2)) * T, (sigma1 * np.sqrt(T)))
            eta2 = np.divide(np.log(spot2 / K) + (r - 0.5 * (sigma2 ** 2)) * T, (sigma2 * np.sqrt(T)))

            beta1 = np.divide(np.log(spot1 / spot2) - 0.5 * ((sig ** 2) * np.sqrt(T)), np.sqrt(sig) * np.sqrt(T))
            beta2 = np.divide(np.log(spot2 / spot1) - 0.5 * ((sig ** 2) * np.sqrt(T)), np.sqrt(sig) * np.sqrt(T))

            gamma1 = eta1 - sigma1 * np.sqrt(T)
            gamma2 = eta2 - sigma2 * np.sqrt(T)

        for i in range(0,N-1):

            for j in range(0,nbPaths):

                price[j,i] = spot1[j,i] * bivnormcdf(eta1[j,i],beta1[j,i],rho_indiv1) + \
                             spot2[j,i] * bivnormcdf(eta2[j,i],beta2[j,i],rho_indiv2) - \
                             K * np.exp(-r * T[j,i]) * bivnormcdf(gamma1[j,i],gamma2[j,i],rho)

        return price

    def get_Stulz_delta(self, S=None, cov=None, r=None, K=None, matu=None, N=None, nbPaths=None):

        dt = matu / N
        T = np.arange(0, N) * dt
        T = np.repeat(np.flip(T[None, :]), S.shape[0], axis=0)

        spot1 = np.zeros((nbPaths, N))
        spot2 = np.zeros((nbPaths, N))

        eta1 = np.zeros((nbPaths, N))
        eta2 = np.zeros((nbPaths, N))

        beta1 = np.zeros((nbPaths, N))
        beta2 = np.zeros((nbPaths, N))

        gamma1 = np.zeros((nbPaths, N))
        gamma2 = np.zeros((nbPaths, N))

        spot1[:, :] = S[:, :, 0]
        spot2[:, :] = S[:, :, 1]

        sigma1 = np.sqrt(cov[0, 0])
        sigma2 = np.sqrt(cov[1, 1])
        rho = cov[0, 1] / (sigma1 * sigma2)

        sig = cov[0,0] + cov[1,1] - 2 * cov[0,1]
        rho_indiv1 = (rho * sigma2 - sigma1) / np.sqrt(sig)
        rho_indiv2 = (rho * sigma1 - sigma2) / np.sqrt(sig)

        delta1 = np.zeros((nbPaths, N-1))
        delta2 = np.zeros((nbPaths, N-1))

        with np.errstate(divide="ignore"):
            eta1 = np.divide(np.log(spot1 / K) + (r + 0.5 * (sigma1 ** 2)) * T, (sigma1 * np.sqrt(T)))
            eta2 = np.divide(np.log(spot2 / K) + (r - 0.5 * (sigma2 ** 2)) * T, (sigma2 * np.sqrt(T)))

            beta1 = np.divide(np.log(spot1 / spot2) - 0.5 * ((sig ** 2) * np.sqrt(T)), np.sqrt(sig) * np.sqrt(T))
            beta2 = np.divide(np.log(spot2 / spot1) - 0.5 * ((sig ** 2) * np.sqrt(T)), np.sqrt(sig) * np.sqrt(T))

            gamma1 = eta1 - sigma1 * np.sqrt(T)
            gamma2 = eta2 - sigma2 * np.sqrt(T)

        for i in range(0,N-1):

            for j in range(0, nbPaths):

                delta1[j,i] = bivnormcdf(eta1[j,i],beta1[j,i],rho_indiv1)+ \
                         stats.norm.cdf(np.divide(eta1[j,i]-rho_indiv1*beta1[j,i],np.sqrt(1-rho_indiv1**2)))* \
                         np.exp(-0.5*beta1[j,i]*beta1[j,i])*(1/np.sqrt(np.min(np.log((spot1),np.log(spot2)))))- \
                         np.divide(spot2[j,i],spot1[j,i])*stats.norm.cdf(np.divide(eta2[j,i]-rho_indiv2*beta2[j,i], \
                         np.sqrt(1-rho_indiv2**2)))*np.exp(-0.5*beta2[j,i]*beta2[j,i])* \
                         (1 / np.sqrt(np.min(np.log((spot1), np.log(spot2)))))

                delta2[j,i] = bivnormcdf(eta2[j,i],beta2[j,i],rho_indiv2)+ \
                         stats.norm.cdf(np.divide(eta2[j,i]-rho_indiv2*beta2[j,i],np.sqrt(1-rho_indiv2**2)))* \
                         np.exp(-0.5*beta2[j,i]*beta2[j,i])*(1/np.sqrt(np.min(np.log((spot1),np.log(spot2)))))- \
                         np.divide(spot1[j,i],spot2[j,i])*stats.norm.cdf(np.divide(eta1[j,i]-rho_indiv1*beta1[j,i], \
                         np.sqrt(1-rho_indiv1**2)))*np.exp(-0.5*beta1[j,i]*beta1[j,i])* \
                         (1 / np.sqrt(np.min(np.log((spot1), np.log(spot2)))))

        return delta1, delta2

m = np.array([0.1, 0.1])
spot_init = np.array([100, 100])
T = 1
N = 30
cov = np.array([[0.1, 0.05], [0.05, 0.1]])
prob1 = 0.3
prob2 = 0.3
r = 0.0
K = 100
matu = 1
nbPaths = 1

test = multiJumpDiffusion(m=m, spot_init=spot_init, T=T, N=N, cov=cov, prob1=prob1, prob2=prob2)
S = test.gen_path(nbPaths=nbPaths)

rainbow = EuropeanRainbow()
price = rainbow.get_Stulz_price(S=S, cov=cov, r=r, K=K, matu=matu, N=N, nbPaths=nbPaths)
delta = rainbow.get_Stulz_delta(S=S, cov=cov, r=r, K=K, matu=matu, N=N, nbPaths=nbPaths)

# price and delta coded
# we still need to code the pnl part
# the results are super weird, we strongly believe there is a mistake in the formula we wrote
