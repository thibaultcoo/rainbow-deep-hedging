#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.stats as stats

from dynamics.singleGeometricBrownian import GeometricBrownianMotion

class EuropeanPut:
    def __init__(self):
        pass
    
    # we will use numpy to generate everything
    # i believe all return vectors
    
    # S -> spot price
    # sigma -> constant volatility
    # r -> constant risk-free rate
    # q -> constant dividend rate
    # K -> strike price
    # matu -> maturity of the option
    # N -> number of steps until maturity
    
    def get_BS_price(self, S = None, sigma = None, r = None, K = None, matu = None, N = None):
        
        dt = np.divide(matu,N)
        T = np.arange(0, N)*dt
        T = np.repeat(np.flip(T[None,:]), S.shape[0], axis=0)
        
        with np.errstate(divide="ignore"):
            d1 = np.divide(np.log(S/K) + (r+0.5*sigma**2) * T, sigma * np.sqrt(T))
            d2 = np.divide(np.log(S/K) + (r-0.5*sigma**2) * T, sigma * np.sqrt(T))
        
        return (K * np.exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0) - S * stats.norm.cdf(-d1, 0.0, 1.0))
    
    def get_BS_delta(self, S = None, sigma = None, r = None, K = None, matu = None, N = None):
        
        dt = np.divide(matu,N)
        T = np.arange(0, N)*dt
        T = np.repeat(np.flip(T[None,:]), S.shape[0], axis=0)

        with np.errstate(divide="ignore"):
            d1 = np.divide(np.log(S/K) + (r+0.5*sigma**2) * T, sigma * np.sqrt(T))
        
        return stats.norm.cdf(-d1, 0.0, 1.0) - 1
    
    def get_BS_PnL(self, S = None, payoff = None, delta = None, matu = None, r = None, final_period_cost = None, eps = None, N = None):
        
        dt = np.divide(matu,N)
        
        # we compute the initial PnL
        PnL_BS = np.multiply(S[:,0], - delta[:,0])
        PnL_BS = PnL_BS - np.abs(delta[:,0]) * S[:,0] * eps
        PnL_BS = PnL_BS * np.exp(r * dt)
        
        # we compute the PnL at each in-between time steps
        for t in range(1, N):
            PnL_BS = PnL_BS + np.multiply(S[:,t], -delta[:,t] + delta[:,t-1])
            PnL_BS = PnL_BS - np.abs(delta[:,t] - delta[:,t-1]) * S[:,t] * eps
            PnL_BS = PnL_BS * np.exp(r * dt)
        
        # we compute the final PnL
        PnL_BS = PnL_BS + np.multiply(S[:,N-1], delta[:,N-1]) + payoff
        
        if final_period_cost:
            PnL_BS = PnL_BS - np.abs(delta[:,N-1]) * S[:,N-1] * eps
            
        return PnL_BS

#put = EuropeanPut()
#S = GeometricBrownianMotion(s0=100, sigma = 0.2, r = 0.0, matu = 1, N = 30).gen_path(10)
#price = put.get_BS_price(S = S, sigma = 0.2, r = 0.0, K = 100, matu = 1, N = 30)

#print(price)