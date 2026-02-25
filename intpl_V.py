# Author: Azar Shakoori, Winter 26 (Len)
# intpl(x,y): compute interpolation polynomial coefficients (Vandermonde approach)
import numpy as np
import numpy.linalg
def intpl(x,y):
    n = numpy.shape(x)[0]-1
    V = numpy.zeros((n+1,n+1))
    for i in range(0,n+1):
        V[i,0] = 1.0
        for j in range(1,n+1):
            V[i,j] = V[i,j-1] * x[i]
    print("Cond. nr. is "+str(numpy.linalg.cond(V))+" for n="+str(n))
    a = numpy.linalg.solve(V,y)
    return a