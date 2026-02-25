# Author: Azar Shakoori Winter 26
# intpl_V Driver
# Input: interpolation nodes
# Output: array a of teh coefficients of a0=a1x+a2x^2+a3x^3+...
import numpy as np
import numpy.linalg
from intpl_V import intpl
x = np.array([0, 1,2,3])
y = np.array([1, 2.71828183, 7.38905610, 20.08553692])
a = intpl(x,y)
print(a)