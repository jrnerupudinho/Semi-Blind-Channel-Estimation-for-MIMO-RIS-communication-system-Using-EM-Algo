import numpy as np

def comm_mat(m,n):
    # determine permutation applied by K
    w = np.arange(m*n).reshape((m,n),order='F').T.ravel(order='F')

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(m*n)[w,:]

n_rx = 8
N = 32
n_tx = 1
c = comm_mat(n_rx*N*n_tx,n_rx*N*n_tx)
print(c.shape)