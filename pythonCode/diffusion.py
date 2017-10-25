#!/usr/bin/python

# Outer code for setting up the diffusion problem on a uniform
# grid and calling the function to perform the diffusion and plot.

from __future__ import absolute_import, division, print_function

import math
import matplotlib.pyplot as plt
import numpy as np

# read in all the linear advection schemes, initial conditions and other
# code associated with this application
execfile("diffusionSchemes.py")
execfile("diagnostics.py")
execfile("initialConditions.py")

def main(xmin = 0., xmax = 1., nx = 41, nt = 40, dt = 0.1, K = 1e-3, squareWaveMin = 0.4, squareWaveMax = 0.6):
    "Diffuse a sqaureWave between squareWaveMin and squareWaveMax on a \
    domain between x-xmin and x = xmax split over nx spatial steps\
    with diffusion coefficient K, time step dt for nt time steps"
    
    #Parameters defined as arguments
    
    #derived parameters
    dx = (xmax - xmin) / (nx-1)
    d = K * dt / dx**2 #nondimensional diffusion coefficient
    
    print("non-dimensional diffusion coefficient = ", d)
    print("dx = ", dx, " dt = ", dt, " nt = ", nt)
    print("end time = ", nt * dt)
    
    #spatial points for plotting and for defining initial conditions
    x = np.zeros(nx)
    
    for j in xrange(nx):
        x[j] = xmin + j*dx
    print( "x = ", x)
    
    #Initial conditions
    phiOld = squareWave(x, squareWaveMin, squareWaveMax)
    
    #analytic solution (of squarewave profile in an infinite domain)
    phiAnalytic = analyticErf(x, K * dt* nt, squareWaveMin, squareWaveMax)
    
    #diffusion using FTCS and BTCS
    #copy allows you to copy by value, not by reference - original does not change
    phiFTCS = FTCS(phiOld.copy(), d, nt)
    phiBTCS = BTCS(phiOld.copy(), d, nt) 
    
    #calculate and print out error norms
    print("FTCS L2 error norm = ", L2ErrorNorm(phiFTCS, phiAnalytic))
    print("BTCS L2 error norm = ", L2ErrorNorm(phiBTCS, phiAnalytic))
    
    #plot the solutions
    
    font = {'size' : 15}
    plt.rc('font', **font)
    plt.figure(1)
    plt.clf()
    #clear figure
    plt.ion()
    #interactive on
    plt.plot(x, phiAnalytic, label = 'Analytic', color = 'black', linestyle = '--', linewidth = 2)
    plt.plot(x,phiFTCS, label = 'FTCS', color = 'blue')
    plt.plot(x, phiBTCS, label = 'BTCS', color = 'red')
    plt.axhline(0, linestyle = ':', color = 'black')
    plt.ylim([0,1])
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.xlabel('$x$')
    plt.savefig('plots/FTCS_BTCS.pdf')
    
    plt.figure(2)
    plt.plot(x, phiFTCS - phiAnalytic, label = 'FTCS', color = 'blue')
    plt.plot(x, phiBTCS - phiAnalytic, label = 'BTCS', color = 'red')
    plt.ylim([-0.005,0.005])
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('$x$')
    plt.ylabel('Error')
    plt.savefig('plots/FTCS_BTCS_Error.pdf')
    
    plt.figure(3)
    plt.plot(x, phiAnalytic, label = 'Analytic', color = 'black', linestyle = '--', linewidth = 2)
    plt.plot(x,phiFTCS, label = 'FTCS', color = 'blue')
    plt.axhline(0, linestyle = ':', color = 'black')
    plt.ylim([0,1])
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.xlabel('$x$')
    plt.savefig('plots/FTCS.pdf')
    

dxs=[0.01,0.025,0.04,0.05,0.076,0.11, 0.17]
l2f=[5.9e-4,0.004, 0.0093,0.015,0.027,0.013,0.024]
l2b=[6.8e-4, 0.004,0.0099,0.0149,0.041,0.12,0.16]

gradf = np.zeros(len(dxs)-1)
for i in xrange(0,len(gradf)):
    gradf[i]=(math.log(l2f[i+1])-math.log(l2f[i]))/(math.log(dxs[i+1])-math.log(dxs[i]))
gradb = np.zeros(len(dxs)-1)
for i in xrange(0,len(gradb)):
    gradb[i]=(math.log(l2b[i+1])-math.log(l2b[i]))/(math.log(dxs[i+1])-math.log(dxs[i]))
print(gradf)
print(gradb)


plt.figure(4)
plt.loglog(dxs,l2f, label = 'FTCS', color = 'blue')
plt.loglog(dxs,l2b, label = 'BTCS', color = 'red')
plt.legend(bbox_to_anchor=(1, 1))
plt.xlabel('$\Delta x$')
plt.ylabel('l2 Error')
plt.savefig('plots/FTCS_BTCS_dx_error.pdf')
    