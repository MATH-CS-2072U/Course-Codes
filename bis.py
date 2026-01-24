# Bisection Algorithm, to be used in Tutorial 2, Ontario Tech U, 2026.
# Updated  to use only one function evaluation per iteration.
import numpy as np

# Inputs: function handle f, initial left and right boundary l and r, tolerance for error and residual eps_x and eps_f (all of type float), max number of iterations (positive integer).
def bis(f,l,r,eps_x,eps_f,kMax):
    # Initialize: copy input values, set defualt convergence flag.
    a = l
    b = r
    conv = False
    fl = f(a)                          # Evaluate at the left boundary once before entering the loop.
    dat = []
    # The bisection loop.
    for i in range(kMax):
        m = (a+b)/2.0
        fm = f(m)                                       
        if fm * fl < 0.0:              # Bisection step: discard one half of the domain.
            b = m
        else:
            a = m
            fl = fm                    # If we update the left boundary, we must also update the left function value!
        err = abs(b-a)                 # Upper bound for the error.
        res = abs(f(m))                # Residual.
        dat.append([i,err,res])
        if err < eps_x and res < eps_f:
            conv = True                # If converged, set convergence flag and exit.
            break
    return m, err, res, conv, np.array(dat)
