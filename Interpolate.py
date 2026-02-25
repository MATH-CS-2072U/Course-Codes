# Author: L. van Veen, UOIT, 2018
# Script for tutorial 6 on interpolation
import math
import numpy
import numpy.linalg                      # numpy is almost a subset of scipy, but cond is not included in scipy
import matplotlib.pyplot as plt          # For plotting

# As programmed in lecture 14:
def intpl(x,y):                          # Bad method for computing the Lagrange interpolating polynomial
    n = numpy.shape(x)[0]-1              # Order of the interpolant
    V = numpy.zeros((n+1,n+1))           # Allocate Vandermonde matrix
    for i in range(0,n+1):               # Compute matrix elements
        V[i,0] = 1.0
        for j in range(1,n+1):
            V[i,j] = V[i,j-1] * x[i]
     # Diagnostic output of the condition number
    print("Cond. nr. is "+str(numpy.linalg.cond(V))+" for n="+str(n))
    a = numpy.linalg.solve(V,y)          # Solve for coefficients
    return a

def f(x):                                # Test function
    return math.exp(x)               

def P(x,a):                              # Evaluate interpolant using Horner's algorithm (see lecture 10)
    n = numpy.shape(a)[0]-1
    Q = a[n-1] + a[n] * x
    for k in range(n-2,-1,-1):
        Q = Q * x + a[k]
    return Q

for n in range(4,16):                    # Test for increasing number of interpolation nodes: Stopped here Feb. 25-26
    x = numpy.zeros(n+1);
    y = numpy.zeros(n+1);
    for k in range(0,n+1):               # Compute interpolation data
        x[k] = float(k)
        y[k] = f(x[k])
    a = intpl(x,y)                       # Compute interpolant
    
    np = 1000                            # Plot function and interpolant on np points between 0 and n
    xs = numpy.zeros(np)
    yf = numpy.zeros(np)
    yp = numpy.zeros(np)
    for k in range(0,np):
        xs[k] = float(k)*float(n)/float(np)
        yf[k] = f(xs[k])
        yp[k] = P(xs[k],a)
    plt.plot(xs,yf,'-k',xs,yp,'-r')
    #plt.ylim(0,100)                     # controls controls the vertical axis limits of your plot to be able to see the black curve
    plt.show()
    relerr = 0.0                         # Error of the linear solve due to bad conditioning 
    norm = 0.0
    for k in range(0,n+1):
        relerr = relerr + (f(x[k])-P(x[k],a))**2 # Compare function to interpolant at nodes
        norm = norm + f(x[k])**2
    relerr = math.sqrt(relerr/norm)
    print("Relative error of lin solve="+str(relerr))

