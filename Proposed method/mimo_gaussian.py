#MIMO Gaussian

import numpy as np
from numpy.linalg import norm
from scipy import special as sp
import matplotlib.pyplot as plt
import itertools
from scipy import linalg
import mpmath as mp
import tensorflow as tf

def channelMatrix(n_tx,n_rx,N,varh):
  H_BS = np.random.normal(loc=0, scale=np.sqrt(varh/2), size=(N, n_tx*2)).view(np.complex128)
  H_SU = np.random.normal(loc=0, scale=np.sqrt(varh/2), size=(n_rx, N*2)).view(np.complex128)
  h = linalg.khatri_rao(H_BS.T, H_SU).flatten()
  # print(tf.convert_to_tensor(h))
  return h

def symbols(n_tx,varx,T_d):
  return np.random.normal(loc=0, scale=np.sqrt(varx/2), size=(T_d, n_tx*2)).view(np.complex128)

def pilotSymbols(n_tx,varx,T_p):
  return np.random.normal(loc=0, scale=np.sqrt(varx/2), size=(T_p, n_tx*2)).view(np.complex128)



def em(y_d,y_p,T_d,T_p,z_p,PsiTilde_td ,all_possibleSymbols,M,varn,itera,h_initial):
  n_rx,_ = y_d[0].shape
  theta_t = h_initial
  print("inital theta",norm(theta_t))
  for l in range(itera):
    m_secondTermNumer = np.zeros(((N)*n_tx*n_rx,1),dtype='complex128')
    m_secondTermDenom = np.zeros(((N)*n_tx*n_rx,(N)*n_tx*n_rx),dtype='complex128')
    m_firstTerm = np.zeros((n_rx,n_rx*N*n_tx*n_rx),dtype='complex128')
    m_firstTermInv = np.zeros((N*n_tx*n_rx*n_rx,N*n_tx*n_rx*n_rx),dtype='complex128')
    # beta_denominator = 0 
    for t in range(0,T_d):
      beta_denominator = 0  
      for k in range(0,np.power(M,n_tx)):
        first_dummy_den = Y_d[t]-np.matmul(np.kron(np.kron(PsiTilde_td[:,t][np.newaxis],all_possibleSymbols[k][np.newaxis]),np.eye(n_rx,dtype='complex128')),theta_t)
        beta_denominator += mp.exp(-np.power(norm(first_dummy_den),2)/np.power(varn,2))
      for j in range(0,np.power(M,n_tx)):
        first_dummy = Y_d[t]-np.matmul(np.kron(np.kron(PsiTilde_td[:,t][np.newaxis],all_possibleSymbols[j][np.newaxis]),np.eye(n_rx,dtype='complex128')),theta_t)
        beta_exp = mp.exp(-np.power(norm(first_dummy),2)/np.power(varn,2))/beta_denominator
        m_secondTermNumer = np.add(m_secondTermNumer,(beta_exp)*np.matmul(np.conjugate(np.kron(np.kron(PsiTilde_td[:,t][np.newaxis],all_possibleSymbols[j][np.newaxis]),np.eye(n_rx,dtype='complex128'))).T,Y_d[t]))
        m_secondTermDenom = np.add(m_secondTermDenom,(beta_exp)*np.matmul(np.conjugate(np.kron(np.kron(PsiTilde_td[:,t][np.newaxis],all_possibleSymbols[j][np.newaxis]),np.eye(n_rx,dtype='complex128'))).T,(np.kron(np.kron(PsiTilde_td[:,t][np.newaxis],all_possibleSymbols[j][np.newaxis]),np.eye(n_rx,dtype='complex128')))))
    for t in range(0,T_p):
      m_firstTerm += np.matmul(y_p[t],np.conjugate(z_p[t]).T)
      m_firstTermInv += np.matmul(z_p[t],np.conjugate(z_p[t]).T)
    m_secondTermDenom = helperPrecisionFloat(m_secondTermDenom)
    m_secondTermNumer = helperPrecisionFloat(m_secondTermNumer)
    theta_tPlusOne = np.linalg.solve(m_firstTerm + m_secondTermDenom,m_firstTermInv + m_secondTermNumer)
    theta_t = theta_tPlusOne
    print(norm(theta_t))
  return theta_t


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

def helperPrecisionFloat(matrix):
    rows,cols = matrix.shape
    a = np.zeros((rows,cols),dtype = "complex128")
    for i in range(rows):
        for j in range(cols):
            a[i,j] = complex(float(np.real(matrix[i,j])),float(np.imag(matrix[i,j])))
    return a

    
def receivedSignals(T_p,T_d,PsiTilde_tp,PsiTilde_td ,n_rx,n_tx,X_d,X_p,h,varn):
  z_p = []
  z_d = []
  y_p = []
  y_d = []
  H = np.kron(h[:,np.newaxis].T,np.eye(n_rx,dtype='complex128'))
  for i in range(T_p):
    z_pt = np.kron(np.kron(PsiTilde_tp[:,i].T,X_p[i].T),np.eye(n_rx,dtype='complex128')).flatten()
    z_p.append(z_pt[:,np.newaxis])
    n_pt = np.random.normal(loc=0, scale=np.sqrt(varn/2), size=(n_rx, 1*2)).view(np.complex128) #Pilot noise
    y_p.append(np.matmul(H,z_pt)[:,np.newaxis] + n_pt) #creates a list with arrays
  for j in range(T_d):
    z_dt = np.kron(np.kron(PsiTilde_td[:,j].T,X_d[j].T),np.eye(n_rx,dtype='complex128')).flatten()
    z_d.append(z_dt[:,np.newaxis])
    n_dt = np.random.normal(loc=0, scale=np.sqrt(varn/2), size=(n_rx,1*2)).view(np.complex128) #Data noise
    y_d.append(np.matmul(H,z_dt)[:,np.newaxis] + n_dt) #creates a list with arrays
  h_initial = np.matmul(np.hstack(y_p),np.linalg.pinv(np.hstack(z_p)))
  return y_p,y_d,z_p,z_d,h_initial


T_d = 50; # Number of data symbols
T_p = [4,8,12,16,20,24,28,32,36,40]; # Number of pilot symbols
# T_p = [4]
N = 32; # Number of IRS elements
n_rx = 8; # Number of receive antennas
n_tx = 1; # Number of tr ansmit antennas
itera = 5; # Number of EM iterations
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
mse = np.zeros((monte_iter,len(T_p)))
for i in range(monte_iter):
    h = channelMatrix(n_tx,n_rx,N,varh)
    X_d = symbols(n_tx,varx,T_d)
    for tp in range(len(T_p)):      
      PsiTilde_tp,PsiTilde_td = irsMatrix(T_p[tp],T_d,N,beta_min,amp)
      # print(np.matmul(PsiTilde_tp,np.conjugate(PsiTilde_tp).T))
      X_p = pilotSymbols(n_tx,varx,T_p[tp])
      y_p,y_d,z_p,z_d,h_initial = receivedSignals(T_p[tp],T_d,PsiTilde_tp,PsiTilde_td ,n_rx,n_tx,X_d,X_p,h,varn)
      theta_hat = em(y_d,y_p,T_d,T_p[tp],z_p,PsiTilde_td,M_symbols,varn,itera,h_initial)
      print("original:",norm(h))
      print("predicted",norm(theta_hat))
      mse[i][tp] = np.matrix.trace(np.abs(np.matmul((np.conjugate(theta_hat-h[:,np.newaxis]).T),(theta_hat-h[:,np.newaxis]))))/np.power(norm(h[:,np.newaxis]),2)
      # mse[i][tp] = np.power(norm(np.subtract(theta_hat,h[:,np.newaxis])),2)/np.power(norm(h[:,np.newaxis]),2)
    if(i%5 == 0):
      print("The monte carlo Iteration Number is: ",i)
mse = np.average(mse, axis=0)

plt.plot(T_p, mse, label = 'Proposed method')
plt.grid(color = 'red', linestyle = '--', linewidth = 0.5)
plt.ylabel('NMSE')
plt.xlabel('T_p')
plt.xticks(T_p)
plt.yscale("log")
plt.title(' Propsed methond with DFT for pilots')
plt.show()




