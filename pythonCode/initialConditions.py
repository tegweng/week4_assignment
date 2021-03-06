import numpy as np

# Initial conditions function for diffusion

def squareWave(x,alpha,beta):
    "A square wave as a function of position, x, which is 1 between alpha"
    "and beta and zero elsewhere. The initialisation is conservative so"
    "that each phi contains the correct quantity integrated over a region"
    "a distance dx/2 either side of x"
    
    phi = np.zeros_like(x)
    
    # The grid spacing (assumed uniform)
    dx = x[1] - x[0]
    
    # Set phi away from the end points (assume zero at the end points)
    for j in xrange(1,len(x)-1):
        # edges of the grid box (using west and east notation)
        xw = x[j] - 0.5*dx
        xe = x[j] + 0.5*dx
        
        #integral quantity of phi
        phi[j] = max((min(beta, xe) - max(alpha, xw))/dx, 0)

    return phi

def sample(x, alpha, beta):
    #function which samples the function rather than checking the integration bounds
    #likely to over or under estimate the area under the curve 
    if float(alpha) >= float(beta) :
        raise ValueError('Error: squarewaveMin must be smaller than squarewaveMax')
    
    phi = np.zeros_like(x)
    
    dx= x[1]-x[0]
    
    for j in xrange(1, len(x)-1):
        if j < (float(alpha) / float(dx)) or j > (float(beta) / float(dx)):
            phi[j] = 0
        else: 
            phi[j] = 1
    return phi
    
try:
    sample([0,2], 3,2)
except ValueError:
    pass
else:
    print('Error in sample function, Min must be smaller than Max')
    
