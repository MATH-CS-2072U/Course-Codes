import numpy as np
small = 1e-14

# Swap two rows od an array:
def swap(A,p,q):
    dum = np.copy(A[p-1,:])
    A[p-1,:] = np.copy(A[q-1,:])
    A[q-1,:] = np.copy(dum)
    return A
    
# Note: indices are math indices, starting from 1!
def Gauss_elim(A):
    n = np.shape(A)[0]
    B = np.copy(A)
    success = True
    for j in range(1,n):
        for i in range(j+1,n+1):
            p = np.argmax(abs(B[i-1:n+1,j-1])) + j
            B = swap(B,i,p)
            if abs(B[j-1,j]) < small:
                print('There may not be a solution!')
                success = False
                return B,success
            m = B[i-1,j-1]/B[j-1,j-1]
            B[i-1,:] = B[i-1,:] - m * B[j-1,:]
    return B,success

a = np.array([[2,2,-0.5,2],[2,2,2,-2],[2,1,0,4]])
b = Gauss_elim(a)
print(b)
