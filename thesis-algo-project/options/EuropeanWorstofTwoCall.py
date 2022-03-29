import numpy as np
import scipy.stats as stats

from dynamics.rainbowDyn import multiGeomBrownian
from functions.bivariateNormal import bivnormcdf

class EuropeanWorstofTwoCall:
    def __init__(self):
        pass

    def get_Stulz_price(self, S=None, cov=None, r=None, K=None, matu=None,
                        N=None, nbPaths=None, dt=None):

        T = np.arange(0, (N + 1)) * dt
        T = np.repeat(np.flip(T[None, :]), S.shape[0], axis=0)

        spot1 = np.zeros((nbPaths, N + 1))
        spot2 = np.zeros((nbPaths, N + 1))

        eta1 = np.zeros((nbPaths, N + 1))
        eta2 = np.zeros((nbPaths, N + 1))

        beta1 = np.zeros((nbPaths, N + 1))
        beta2 = np.zeros((nbPaths, N + 1))

        gamma1 = np.zeros((nbPaths, N + 1))
        gamma2 = np.zeros((nbPaths, N + 1))

        spot1[:, :] = S[:, :, 0]
        spot2[:, :] = S[:, :, 1]

        sigma1 = np.sqrt(cov[0, 0])
        sigma2 = np.sqrt(cov[1, 1])
        rho = cov[0, 1] / (sigma1 * sigma2)

        sig = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
        rho_indiv1 = (rho * sigma2 - sigma1) / np.sqrt(sig)
        rho_indiv2 = (rho * sigma1 - sigma2) / np.sqrt(sig)

        price = np.zeros((nbPaths, N + 1))

        with np.errstate(divide="ignore"):
            eta1 = np.divide(np.log(spot1 / K) +
                   (r + 0.5 * (sigma1 ** 2)) * T, (sigma1 * np.sqrt(T)))

            eta2 = np.divide(np.log(spot2 / K) +
                   (r + 0.5 * (sigma2 ** 2)) * T, (sigma2 * np.sqrt(T)))

            beta1 = np.divide(np.log(spot2 / spot1) -
                    0.5 * ((sig ** 2) * np.sqrt(T)), np.sqrt(sig) * np.sqrt(T))

            beta2 = np.divide(np.log(spot1 / spot2) -
                    0.5 * ((sig ** 2) * np.sqrt(T)), np.sqrt(sig) * np.sqrt(T))

            gamma1 = eta1 - sigma1 * np.sqrt(T)
            gamma2 = eta2 - sigma2 * np.sqrt(T)

        for i in range(0, N):

            for j in range(0, nbPaths):

                price[j, i] = spot2[j, i] * bivnormcdf(eta2[j, i], beta2[j, i], rho_indiv2) + \
                              spot1[j, i] * bivnormcdf(eta1[j, i], beta1[j, i], rho_indiv1) - \
                              K * np.exp(-r * T[j, i]) * bivnormcdf(gamma2[j, i], gamma1[j, i], rho)

                if price[j, i] < 0:
                    price[j, i] = 0
                    
            price[:,N] = np.maximum(0, np.minimum(spot1[:,N]-K, spot2[:,N]-K))

        return price

    def get_Stulz_delta(self, S=None, cov=None, r=None, K=None, matu=None,
                            N=None, nbPaths=None, dt=None):

            T = np.arange(0, (N + 1)) * dt
            T = np.repeat(np.flip(T[None, :]), S.shape[0], axis=0)

            spot1 = np.zeros((nbPaths, N + 1))
            spot2 = np.zeros((nbPaths, N + 1))

            eta1 = np.zeros((nbPaths, N + 1))
            eta2 = np.zeros((nbPaths, N + 1))

            beta1 = np.zeros((nbPaths, N + 1))
            beta2 = np.zeros((nbPaths, N + 1))

            gamma1 = np.zeros((nbPaths, N + 1))
            gamma2 = np.zeros((nbPaths, N + 1))

            spot1[:, :] = S[:, :, 0]
            spot2[:, :] = S[:, :, 1]

            sigma1 = np.sqrt(cov[0, 0])
            sigma2 = np.sqrt(cov[1, 1])
            rho = cov[0, 1] / (sigma1 * sigma2)

            sig = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
            rho_indiv1 = (rho * sigma2 - sigma1) / np.sqrt(sig)
            rho_indiv2 = (rho * sigma1 - sigma2) / np.sqrt(sig)

            delta1 = np.zeros((nbPaths, N + 1))
            delta2 = np.zeros((nbPaths, N + 1))

            with np.errstate(divide="ignore"):
                eta1 = np.divide(np.log(spot1 / K) + (r + 0.5 * (sigma1 ** 2)) * T, (sigma1 * np.sqrt(T)))
                eta2 = np.divide(np.log(spot2 / K) + (r + 0.5 * (sigma2 ** 2)) * T, (sigma2 * np.sqrt(T)))

                beta1 = np.divide(np.log(spot2 / spot1) - 0.5 * ((sig ** 2) * np.sqrt(T)), np.sqrt(sig) * np.sqrt(T))
                beta2 = np.divide(np.log(spot1 / spot2) - 0.5 * ((sig ** 2) * np.sqrt(T)), np.sqrt(sig) * np.sqrt(T))

                gamma1 = eta1 - sigma1 * np.sqrt(T)
                gamma2 = eta2 - sigma2 * np.sqrt(T)

            for i in range(0, N):

                for j in range(0, nbPaths):
                    
                    delta1[j, i] = bivnormcdf(eta1[j, i], beta1[j, i], rho_indiv1) + \
                                   stats.norm.cdf(np.divide(eta1[j, i] - rho_indiv1 * beta1[j, i], np.sqrt(1 - rho_indiv1 ** 2))) * \
                                   np.exp(-0.5 * beta1[j, i] * beta1[j, i]) * (1 / np.sqrt(np.min(np.log((spot1), np.log(spot2))))) - \
                                   np.divide(spot2[j, i], spot1[j, i]) * stats.norm.cdf(np.divide(eta2[j, i] - rho_indiv2 * beta2[j, i], \
                                   np.sqrt(1 - rho_indiv2 ** 2))) * np.exp(-0.5 * beta2[j, i] * beta2[j, i]) * \
                                   (1 / np.sqrt(np.min(np.log((spot1), np.log(spot2)))))

                    delta2[j, i] = bivnormcdf(eta2[j, i], beta2[j, i], rho_indiv2) + \
                                   stats.norm.cdf(np.divide(eta2[j, i] - rho_indiv2 * beta2[j, i], np.sqrt(1 - rho_indiv2 ** 2))) * \
                                   np.exp(-0.5 * beta1[j, i] * beta1[j, i]) * (1 / np.sqrt(np.min(np.log((spot1), np.log(spot2))))) - \
                                   np.divide(spot1[j, i], spot2[j, i]) * stats.norm.cdf(np.divide(eta1[j, i] - rho_indiv1 * beta1[j, i], \
                                   np.sqrt(1 - rho_indiv1 ** 2))) * np.exp(-0.5 * beta2[j, i] * beta2[j, i]) * \
                                   (1 / np.sqrt(np.min(np.log((spot1), np.log(spot2)))))
 
            # can we assume that the expiry delta of a rainbow option is the vanilla delta considering the worst ssj ?
            # that seems very much correct! so we will implement that little tweak in order to have a final delta!

            delta = np.zeros(((2, nbPaths, N + 1)))
            
            if np.minimum(spot1[:,N], spot2[:,N]) == spot1[:,N]:
                if spot1[:,N] > K:
                    delta1[:,N] = 1
            
            elif np.minimum(spot1[:,N], spot2[:,N]) == spot2[:,N]:
                if spot2[:,N] > K:
                    delta2[:,N] = 1    
                
            delta[0, :, :] = delta1[:, :]
            delta[1, :, :] = delta2[:, :]
            
            return delta

    def get_Stulz_PnL(self, S=None, payoff=None, delta=None, matu=None, r=None, \
                      final_period_cost=None, eps=None, N=None):

        # we compute the initial PnL
        PnL_Stulz = np.multiply(S[:, 0, 0], - delta[0, :, 0]) + np.multiply(S[:, 0, 1], - delta[1, :, 0])
        PnL_Stulz -= np.abs(delta[0, :, 0]) * S[:, 0, 0] * eps - np.abs(delta[1, :, 0]) * S[:, 0, 1] * eps
        PnL_Stulz *= np.exp(r * dt)

        # we compute the PnL at each in-between time steps
        for t in range(1, N):
            PnL_Stulz += + np.multiply(S[:, t, 0], -delta[0, :, t] + delta[0, :, t - 1]) + \
                        np.multiply(S[:, t, 1], -delta[1, :, t] + delta[1, :, t - 1])

            PnL_Stulz += - np.abs(delta[0, :, t] - delta[0, :, t - 1]) * S[:, t, 0] * eps \
                        - np.abs(delta[1, :, t] - delta[1, :, t - 1]) * S[:, t, 1] * eps

            PnL_Stulz *= np.exp(r * dt)

        # we compute the final PnL
        PnL_Stulz += + np.multiply(S[:, N - 1, 0], delta[0, :, N - 2]) + np.multiply(S[:, N - 1, 1],delta[1, :,N - 2]) + payoff

        return PnL_Stulz

cov = np.zeros((2,2))
r = 0.0
K = 100
matu = 1
N = 30
nbPaths = 2
dt = 1/365

cov[0,0] = 0.1
cov[1,1] = 0.08
cov[0,1] = 0.04
cov[1,0] = 0.04

S = multiGeometric(s0=(100,100),T=matu,N=N,cov=cov,dt=dt).gen_path(nbPaths=nbPaths)
        
option = EuropeanWorstofTwoCall()
price = option.get_Stulz_price(S=S, cov=cov, r=r, K=K, matu=matu, N=N, nbPaths=nbPaths, dt=dt)

# price and delta are working perfectly!
# pnl might need a tiny bit rework
