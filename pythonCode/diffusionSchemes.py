# Numerical schemes for simulating diffusion for outer code diffusion.py

from __future__ import absolute_import, division, print_function
import numpy as np

# The linear algebra package for BTCS (for solving the matrix equation)
import scipy.linalg as la

def FTCS(phiOld, d, nt):
    "Diffusion of profile in PhiOld using STCS using non-dimensional \
    diffusion coefficient, d"
    
    nx = len(phiOld)
    
    #new time-step array for phi
    
    phi = phiOld.copy() 
    
    #FTCS for all time steps
    for it in xrange(int(nt)):
        for i in xrange(int(nx-1)):
            phi[i] = phiOld[i] + d * (phiOld[i+1] - 2 * phiOld[i] + phiOld[i-1])
            #Boundary conditions for xmin and xmax
            phi[0]=phi[1]
            phi[-1]=phi[-2] #wraps around to back
        #at the end of each time step we need to set phiOld to phi
        phiOld = phi.copy()
    
    return phi

def BTCS(phi, d, nt):
    "Diffusion of profile in phi using BTCS using non dimensional \
    diffusion coefficient, d, assuming fixed value boundary conditions"
    
    nx = len(phi)
    
    #array representing BTCS
    M=np.zeros([nx,nx])
    
    #zero gradient boundary conditions
    
    M[0,0] = 1.
    M[0,1] = -1.
    M[-1,-1] = 1.
    M[-1,-2] = -1.
    
    for i in xrange(1,nx-1):
        M[i,i-1] = -d
        M[i,i] = 1+2*d
        M[i,i+1] = -d
    
    #BTCS for all time steps
    
    for it in xrange(int(nt)):
        #RHS for zero gradient boundary conditions - have to keep resetting the \
        #boundary to be zero each time step
        phi[0] = 0
        phi[-1] = 0
        #as have to do the following for all time steps it is in the for loop
        phi = la.solve(M, phi)
    
    return phi