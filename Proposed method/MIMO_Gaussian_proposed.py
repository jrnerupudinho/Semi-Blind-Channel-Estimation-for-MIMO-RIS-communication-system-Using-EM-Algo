import numpy as np
import QAM as qp
from numpy.linalg import norm
import matplotlib.pyplot as plt
import itertools
import gmpy2 as gp
from scipy import linalg 

def channelMatrix1(varh,N,n_rx,n_tx):
  H_BS = np.random.normal(loc=0, scale=np.sqrt(varh/2), size=(N, n_tx*2)).view(np.complex128)
  H_SU = np.random.normal(loc=0, scale=np.sqrt(varh/2), size=(n_rx, N*2)).view(np.complex128)
  h_proposed = linalg.khatri_rao(H_BS.T, H_SU).flatten(order='F')
  H = np.kron(h_proposed[:,np.newaxis].T,np.eye(n_rx,dtype= 'complex128'))
  return H


def symbols(n_tx,T_d,varx):
    return np.random.normal(loc=0, scale=np.sqrt(varx/2), size=(n_tx,T_d*2)).view(np.complex128)


def pilotSymbols(n_tx,T_p,varx):
    return np.random.normal(loc=0, scale=np.sqrt(varx/2), size=(n_tx,T_p*2)).view(np.complex128)


def comm_mat(m, n):
    # determine permutation applied by K
    w = np.arange(m * n).reshape((m, n), order="F").T.ravel(order="F")

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(m * n,dtype='complex128')[w, :]


def variance_z(t,n_tx,n_rx,varx,PsiTilde_td):
    N,_ = PsiTilde_td.shape
    P = np.kron(np.kron(np.eye(N,dtype= 'complex128'),comm_mat(1, n_tx*n_rx)),np.eye(n_rx,dtype= 'complex128'))
    Q = np.kron(np.kron(np.eye(N*n_tx,dtype= 'complex128'),comm_mat(1, n_rx)),np.eye(n_rx,dtype= 'complex128'))
    Psi_prod = PsiTilde_td[:,t][:,np.newaxis]@np.conjugate(PsiTilde_td[:,t][:,np.newaxis]).T
    eye_prod = np.eye(n_rx,dtype= 'complex128').flatten(order='F')[:,np.newaxis]@np.conjugate(np.eye(n_rx,dtype= 'complex128').flatten(order='F'))[:,np.newaxis].T
    middle_prod = np.kron(np.kron(Psi_prod,np.eye(n_tx,dtype= 'complex128')),eye_prod)
    last_prod = np.conjugate(Q).T@np.conjugate(P).T
    first_prod = P@Q
    first_middle = first_prod@middle_prod
    rows,cols =     first_middle.shape
    sigma = np.power(varx,2)*(first_middle@last_prod) 
    return sigma

def inv(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]


def EM_Gaussian_proposed(y_d,y_p,T_d,T_p,z_p,PsiTilde_td,varn,itera,H_initial,varx,n_tx):  
    N,_ = PsiTilde_td.shape
    rows, cols = z_p[0].shape
    n_rx, _ = y_d[0].shape
    H_l = H_initial
    j = 0
    while j <= itera:
        h = []
        first_term_inv = np.zeros((rows,rows),dtype = 'complex128')
        second_term_inv = np.zeros((rows,rows),dtype = 'complex128')
        conditional_mean = np.zeros((rows,T_d),dtype = 'complex128')
        for t1 in range(T_p):
            first_term_inv += z_p[t1]@np.conjugate(z_p)[t1].T
        for t2 in range(T_d):
            sigma_t = variance_z(t2,n_tx,n_rx,varx,PsiTilde_td)
            middle_term = np.power(varn,2)*np.eye((n_rx),dtype='complex128') + H_l@sigma_t@np.conjugate(H_l).T
            conditional_mean[:,t2][:,np.newaxis] = sigma_t@np.conjugate(H_l).T@inv(middle_term)@y_d[t2]
            mean_prod =  conditional_mean[:,t2]@np.conjugate( conditional_mean[:,t2]).T
            second_term_var = sigma_t@np.conjugate(H_l).T@inv(middle_term)@H_l@sigma_t
            covar = sigma_t - second_term_var +mean_prod
            second_term_inv += covar
        for r in range(n_rx):
            first_term = np.zeros((1,rows),dtype = 'complex128')
            second_term = np.zeros((1,rows),dtype = 'complex128')
            for t1 in range(T_p):
                first_term += y_p[t1][r,:]*np.conjugate(z_p[t1]).T
            for t2 in range(T_d):
                second_term += y_d[t2][r,:]*np.conjugate(conditional_mean[:,t2]).T
            h.append(np.conjugate(np.matmul((first_term + second_term),inv(first_term_inv + second_term_inv))))
        H_lPlusOne = np.row_stack(np.conjugate(h).T).T
        H_l = H_lPlusOne
        j = j+1
        print(norm(H_l))
    return H_l

def irsMatrix(T_p,T_d,N,beta_min,amp):
  PsiTilde_tp = np.zeros((N,T_p),dtype = 'complex128')
  PsiTilde_td =  []
  for n in range(0,N):
    for t in range(0,T_p):
            PsiTilde_tp[n,t] = np.exp((-1j*2*np.pi*(t)*(n))/(N))
  for t in range(0,T_d):
      beta = (beta_max-beta_min)*np.random.uniform(0,1,(N,1))+beta_min; 
      psi_t = amp*np.exp(1j*beta);
      PsiTilde_td.append(psi_t)  
  PsiTilde_td =np.concatenate( PsiTilde_td, axis=1 )
  return PsiTilde_tp, PsiTilde_td

def helperMPCtoFLoat(matrix):
    rows,cols = matrix.shape
    a = np.zeros((rows,cols),dtype = "complex128")
    for i in range(rows):
        for j in range(cols):
            a[i,j] = complex(float(matrix[i,j].real),float(matrix[i,j].imag))
    return a   

def helperPrecisionFloat(matrix):
    rows,cols = matrix.shape
    a = np.zeros((rows,cols),dtype = "complex128")
    for i in range(rows):
        for j in range(cols):
            a[i,j] = complex(float(np.real(matrix[i,j])),float(np.imag(matrix[i,j])))
    return a
    
def received_proposed(T_p,T_d,PsiTilde_tp,PsiTilde_td ,n_rx,n_tx,X_d,X_p,H,varn):
  z_p = []
  z_d = []
  y_p = []
  y_d = []
  for i in range(T_p):
    z_pt = np.kron(np.kron(PsiTilde_tp[:,i].T,X_p[:,i].T),np.eye(n_rx,dtype='complex128')).flatten(order='F')
    # print(Z_pt.shape)
    z_p.append(z_pt[:,np.newaxis])
    n_pt = np.random.normal(loc=0, scale=np.sqrt(varn/2), size=(n_rx, 1*2)).view(np.complex128) #Pilot noise
    y_p.append(np.matmul(H,z_pt[:,np.newaxis]) + n_pt) #creates a list with arrays
  for j in range(T_d):
    z_dt = np.kron(np.kron(PsiTilde_td[:,j].T,X_d[:,j].T),np.eye(n_rx,dtype='complex128')).flatten(order='F')
    z_d.append(z_dt[:,np.newaxis])
    n_dt = np.random.normal(loc=0, scale=np.sqrt(varn/2), size=(n_rx,1*2)).view(np.complex128) #Data noise
    y_d.append(np.matmul(H,z_dt[:,np.newaxis]) + n_dt)
  h_initial = np.matmul(np.hstack(y_p),np.linalg.pinv(np.hstack(z_p)))
  return y_p,y_d,z_p,z_d,h_initial


T_d = 50; # Number of data symbols
T_p = [8,12,16,20,24,28,32,36,40]; # Number of pilot symbols
N = 32; # Number of IRS elements
n_rx = 2
n_tx = 2; # Number of tr ansmit antennas
itera = 3 ; # Number of EM iterations
monte_iter = 1;  # Number of Monte carlo iterations
varx = 1;  # The variance of the complex data and pilots 
varh = 1; # The variance of the channels
beta_min = 0; #Minimum possible phase of IRS element
beta_max = 2*np.pi; #Maximum possible phase of IRS element
amp = 1 #Constant amplitude: For random amplitudes: np.random.uniform(0,1,(N,1)) 
M_symbols = 4; # Constellation size for data and pilot symbols.
power = 10 #constellation power, needs to be computed from the constellation. 
SNR = [20] #SNR (dB)
varn = 0.1


# #Computation 
MSE_proposed = np.zeros((monte_iter,len(T_p)))
for i in range(monte_iter):
    H = channelMatrix1(varh,N,n_rx,n_tx)
    X_d = symbols(n_tx,T_d,varx)
    for tp in range(len(T_p)):      
      PsiTilde_tp,PsiTilde_td = irsMatrix(T_p[tp],T_d,N,beta_min,amp)
      # PsiTilde_tp = np.insert(PsiTilde_tp,0,np.ones((1,T_p[tp]),dtype='complex128'),axis= 0)
      # print(np.matmul(PsiTilde_tp,np.conjugate(PsiTilde_tp).T))
      X_p = pilotSymbols(n_tx,T_p[tp],varx)
       # x_p = pilotsSymbols_approx(N, M, L_p[tp], n_r, varx) 
      y_p,y_d,z_p,z_d,h_initial = received_proposed(T_p[tp],T_d,PsiTilde_tp,PsiTilde_td ,n_rx,n_tx,X_d,X_p,H,varn)
      H_hat = EM_Gaussian_proposed(y_d,y_p,T_d,T_p[tp],z_p,PsiTilde_td,varn,itera,h_initial,varx,n_tx)
      print("original:",norm(H))
      print("predicted",norm(H_hat))
      MSE_proposed[i][tp] = np.matrix.trace(np.abs(np.matmul((np.conjugate(H_hat-H).T),(H_hat-H))))/np.power(norm(H),2)
      # mse[i][tp] = np.power(norm(np.subtract(theta_hat,h[:,np.newaxis])),2)/np.power(norm(h[:,np.newaxis]),2)
    if(i%5 == 0):
      print("The monte carlo Iteration Number is: ",i)
MSE_proposed = np.average(MSE_proposed, axis=0)
plt.plot(T_p, MSE_proposed, label = 'Proposed method',linestyle='dashed', marker='o')
plt.grid(linestyle = '--', linewidth = 0.5)
plt.ylabel('NMSE')
plt.xlabel('T_p')
plt.xticks(T_p)
plt.yscale("log")
plt.title(' Proposed method and Guassian method - Exact')
plt.legend()
plt.show()



