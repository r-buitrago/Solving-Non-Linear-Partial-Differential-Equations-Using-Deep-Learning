import numpy as np

def randomwalk(xi = 0, t0 = 0., T = 1., sigma = lambda x,t: 2**(1/2), 
               mu = lambda x,t: 0 , M = 1, N = 10):
    #xi is the startingpoint
    #t0 is the initial time
    #T is the terminal time
    #sigma is the random deviation of the PDE (found in the parabolic equation)
    #M is the number of random walks generated
    #N is the number of discretized timesteps
    
    #We will generate values:  
    #W[m,i] = DeltaW_{i}^m = W_{i+1}^m - W_{i}^m;  i = 0, ..., N
    #X[m,i] = X_{i}^m = X^m(t_i);  i = 0, ..., N
    
    timestep = (T-t0)/N 
    t = np.linspace(t0, T, num = N+1)
    #we divide by N so that we have N+1 values between t0 and T (counting both t0 and T) with separation = timestep
    
    #First we generate all the random values that we need (for each path, we need N-1 of them)
    W = np.random.normal( loc = 0., scale = timestep**(1/2), size = (M,N) ) 

    #Now we generated the random values following eq [4] of the article
    X = np.zeros( shape= (M,N+1) )
    X[:,0] = np.full(shape = (M,), fill_value = xi)
    
    sigmavec = np.vectorize(sigma)
    muvec = np.vectorize(mu)

    for i in range(N):
        X[:,i+1] = X[:,i] + muvec(X[:,i],t[i])*timestep + sigmavec(X[:,i],t[i])*W[:,i]
    
    return X, W