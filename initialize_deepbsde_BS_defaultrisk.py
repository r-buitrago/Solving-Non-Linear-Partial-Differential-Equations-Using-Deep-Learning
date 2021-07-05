import numpy as np
import tensorflow as tf
from scipy.stats import norm
import tensorflow_probability as tfp

#In this file we introduce all the parameters from the PDE we want to solve
#These parameters are based on the equation 1 of article "Solving high-dimensional PDE's using Deep Learning" by Jiequn Han et al
#In this particular code, we apply the method to solve the Black-Scholes equation with default risk (model and deduction of PDE available at my thesis)
t0 = 0. #initial time
T = 1. #terminal time
N = 10 #number of discretized time values
xi = 100. #x for which we will find u(t0,xi), ux(t0,xi)
sigmabs = 0.2 #volatility in the Black-Scholes equation
r = 0.02 #interest rate (per annum)
E = 100 #payoff price
g = np.vectorize( lambda x: max(0,x-E) ) #terminal condition
sigma =  lambda x,t: sigmabs * x #sigma from eq. 1 of the article
mu =  lambda x,t: r * x  #mu from eq. 1 of the article

#Here, we introduce the Black-Scholes formula in numpy, which we will use to compare the results of the solution u with the default risk model
def d1(x, t):
    return (np.log(x/E) + (r + 1/2*sigmabs**2)*(T-t))/(sigmabs*(T-t)**(1/2))
def d2(x, t):
    return (np.log(x/E) + (r - 1/2*sigmabs**2)*(T-t))/(sigmabs*(T-t)**(1/2))
def u_true_fun(x, t):
    return x*norm.cdf(d1(x, t)) - E*np.exp(-r*(T-t))*norm.cdf(d2(x, t))
def ux_true_fun(x, t):
    return norm.cdf(d1(x, t)) + 1/( (2*np.pi)**(1/2) * sigmabs * (T - t)**(1/2) )*(np.exp(-d1(x, t)**2/2) - E/x*np.exp(-r*(T-t))*np.exp(-d2(x, t)**2/2) )  

#We define the parameters for the default intensity
delta = 2/3
z = u_true_fun(xi,t0)
gammah = 0.6
gammal = 0.9
beta = 1/2
Q = lambda u, x, t: ( (gammah-gammal) * tf.math.sigmoid(-beta*(u-z)) + gammal ) #default intensity

#Now we introduce the Black-Scholes formula with Tensorflow functions
def d1_tf(x, t):
    return (tf.math.log(x/E) + (r + 1/2*sigmabs**2)*(T-t))/(sigmabs*(T-t)**(1/2))
def d2_tf(x, t):
    return (tf.math.log(x/E) + (r - 1/2*sigmabs**2)*(T-t))/(sigmabs*(T-t)**(1/2))
def v_tf(x, t):
    return x*tfp.distributions.Normal(0,1).cdf(d1_tf(x,t)) - E*np.exp(-r*(T-t))*tfp.distributions.Normal(0,1).cdf(d2_tf(x,t))

f = lambda u,x,t: -(r+Q(u,x,t))*u + delta*Q(u,x,t)*v_tf(x,t) #f from eq. 1 of the article

timestep = (T-t0)/N 
t = np.linspace(t0, T, num = N+1) #discretized time         

print("t0 is equal to", t0)
print("T is equal to", T)
print("N is equal to", N)
print("xi is equal to", xi)
print("timestep is equal to", timestep)
