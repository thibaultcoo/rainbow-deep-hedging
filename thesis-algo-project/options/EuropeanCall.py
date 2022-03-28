import numpy as np
from scipy import stats

class EuropeanCall:
	def __init__(self):
		pass
		
	def call_price(self, S=None, sigma=None, r=None, N=None, K=None, dt=None):

		T = np.arange(0, (N + 1))*dt
		T = np.repeat(np.flip(T[None,:]), S.shape[0], 0)

		with np.errstate(divide='ignore'):
			d1 = np.divide(np.log(S / K) + (r + 0.5 * sigma ** 2) * T, sigma * np.sqrt(T))
			d2 = np.divide(np.log(S / K) + (r -0.5 * sigma ** 2) * T, sigma * np.sqrt(T))
			
		return (S * stats.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
					
	def call_delta(self, S=None, sigma=None, r=None, N=None, K=None, dt=None):

		T = np.arange(0, (N + 1))*dt
		T = np.repeat(np.flip(T[None,:]), S.shape[0], 0)

		with np.errstate(divide='ignore'):
			d1 = np.divide(np.log(S / K) + (r + 0.5 * sigma ** 2) * T, sigma * np.sqrt(T))
			
		return stats.norm.cdf(d1, 0.0, 1.0)

	def call_pnl(self, S=None, payoff=None, delta=None, dt=None, r=None, eps=None):

		N = S.shape[1]-1
		PnL_BS = np.multiply(S[:,0], -delta[:,0])
		PnL_BS -= np.abs(delta[:,0])*S[:,0]*eps
		PnL_BS = PnL_BS*np.exp(r*dt)
		
		for t in range(1, N):
			PnL_BS += np.multiply(S[:,t], -delta[:,t] + delta[:,t-1])
			PnL_BS -= np.abs(delta[:,t] -delta[:,t-1])*S[:,t]*eps
			PnL_BS = PnL_BS*np.exp(r*dt)

		PnL_BS += np.multiply(S[:,N],delta[:,N-1]) + payoff 

		return PnL_BS
