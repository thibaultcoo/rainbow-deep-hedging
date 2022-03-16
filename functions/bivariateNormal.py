import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from math import log, sqrt, exp, pi

# inspired by Hull's paper
def bivnormcdf(a,b,rho):

	if a <= 0. and b <= 0. and rho <= 0. :
		aprime = a/sqrt(2.*(1.-rho**2.))
		bprime = b/sqrt(2.*(1.-rho**2.))
		A = np.array([0.3253030, 0.4211071, 0.1334425, 0.006374323])
		B = np.array([0.1337764, 0.6243247, 1.3425378, 2.2626645])

		t = 0.

		for i in range(4):
			for j in range(4):
				x = B[i]
				y = B[j]
				t += A[i]*A[j]* exp(aprime*(2.*x - aprime) \
				     + (bprime*(2.*y - bprime)) + (2.*rho * (x - aprime)*(y-bprime)))

		p = (sqrt(1.-rho**2.)/pi) * t

	elif a * b * rho <= 0. :
		if a <= 0. and b >= 0. and rho >= 0. :
			p = stats.norm.cdf(a) - bivnormcdf(a,-b,-rho)
		elif a >= 0. and b <= 0. and rho >= 0. :
			p = stats.norm.cdf(b) - bivnormcdf(-a,b,-rho)
		elif a >= 0. and b >= 0. and rho <= 0. :
			p = stats.norm.cdf(a) + stats.norm.cdf(b) - 1. + bivnormcdf(-a,-b,rho)

	elif a*b*rho > 0. :
		if a >= 0. :
			asign = 1.
		else:
			asign = -1.

		if b >= 0.:
			bsign = 1.
		else:
			bsign = -1.

		rho1 = (rho*a - b)*asign/(sqrt(a**2. - (2.*rho*a*b) + b**2.))
		rho2 = (rho*b - a)*bsign/(sqrt(a**2. - (2.*rho*a*b) + b**2.))
		delta = (1. - (asign*bsign))/4.

		p = bivnormcdf(a,0,rho1) + bivnormcdf(b,0,rho2) - delta
	return p