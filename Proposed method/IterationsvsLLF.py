#Iterations vs loglikehood function

import numpy as np
import QAM as qp
from numpy.linalg import norm
from scipy import special as sp
import matplotlib.pyplot as plt
import itertools
from scipy import linalg
import mpmath as mp

def channelMatrix(n_tx,n_rx,N,varh):
  H_BU = np.random.normal(loc=0, scale=np.sqrt(varh/2), size=(n_rx, n_tx*2)).view(np.complex128)  #*2 because we need two values to create a complex number view
  H_BS = np.random.normal(loc=0, scale=np.sqrt(varh/2), size=(N, n_tx*2)).view(np.complex128)
  H_SU = np.random.normal(loc=0, scale=np.sqrt(varh/2), size=(n_rx, N*2)).view(np.complex128)
  h = np.concatenate((H_BU.flatten(), linalg.khatri_rao(H_BS.T, H_SU).flatten()))
  print(norm(h))
  return h

def symbols(n_tx,M,T_d):
  X_d = []
  qam = qp.QAModulation(M)
  qamCons = qam.constellation
  for k in range(T_d):
     X_d.append(np.reshape(qamCons[np.random.choice(range(0, M), n_tx, 'True')],(n_tx,1)))
  args = []
  for i in range(0,n_tx):
    args.append(qamCons)
  all_possibleSymbols = []
  for i in itertools.product(*args):
    all_possibleSymbols.append(i)
  all_possibleSymbols = np.asarray(all_possibleSymbols)
  return X_d,all_possibleSymbols

def pilotSymbols(n_tx,M,T_p):
  X_p = []
  qam = qp.QAModulation(M)
  qamCons = qam.constellation
  for k in range(T_p):
    X_p.append(np.reshape(qamCons[np.random.choice(range(0, M), n_tx, 'True')],(n_tx,1)))
  return X_p



def em(Y_d,Y_p,T_d,T_p,Z_p,Z_d,PsiTilde_td ,all_possibleSymbols,M,varn,itera,h_initial,n_tx):
  n_rx,_ = Y_d[0].shape
  theta_t = h_initial
  print("inital theta",norm(theta_t))
  e_firstTerm =  T_d*n_tx*np.log(M)
  e_secondTerm = (T_d+T_p)*np.log(np.pi*(varn**2))
  logLikelihood = np.zeros((1,itera))
  for l in range(itera):
    m_secondTermNumer = np.zeros(((N+1)*n_tx*n_rx,1),dtype='complex128')
    m_secondTermDenom = np.zeros(((N+1)*n_tx*n_rx,(N+1)*n_tx*n_rx),dtype='complex128')
    m_firstTermNumer = np.zeros(((N+1)*n_tx*n_rx,1),dtype='complex128')
    m_firstTermDenom = np.zeros(((N+1)*n_tx*n_rx,(N+1)*n_tx*n_rx),dtype='complex128')
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
      m_firstTermNumer += np.matmul(np.conjugate(Z_p[t]).T,Y_p[t])
      m_firstTermDenom += np.matmul(np.conjugate(Z_p[t]).T,Z_p[t])
    m_secondTermDenom = helperPrecisionFloat(m_secondTermDenom)
    m_secondTermNumer = helperPrecisionFloat(m_secondTermNumer)
    theta_tPlusOne = np.linalg.solve(m_firstTermDenom + m_secondTermDenom,m_firstTermNumer + m_secondTermNumer)
    theta_t = theta_tPlusOne
    print(norm(theta_t))
    logLikelihood[:,l] = -e_firstTerm-e_secondTerm-((1/varn**2)*norm(Y_p - np.matmul(Z_p,theta_t)))-((1/varn**2)*norm(Y_d - np.matmul(Z_d,theta_t)))
  return theta_t,logLikelihood.reshape(itera,1)

def helperPrecisionFloat(matrix):
    rows,cols = matrix.shape
    a = np.zeros((rows,cols),dtype = "complex128")
    for i in range(rows):
        for j in range(cols):
            a[i,j] = complex(float(np.real(matrix[i,j])),float(np.imag(matrix[i,j])))
    return a

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
    
def receivedSignals(T_p,T_d,PsiTilde_tp,PsiTilde_td ,n_rx,n_tx,X_d,X_p,h,varn,M_symbols,N):
  Z_p = []
  Z_d = []
  Y_p = []
  Y_d = []
  for i in range(T_p):
    Z_pt = np.kron(np.kron(PsiTilde_tp[:,i].T,X_p[i].T),np.eye(n_rx,dtype='complex128'))
    Z_p.append(Z_pt)
    n_pt = np.random.normal(loc=0, scale=np.sqrt(varn/2), size=(n_rx, 1*2)).view(np.complex128) #Pilot noise
    Y_p.append(np.matmul(Z_pt,h[:,np.newaxis]) + n_pt) #creates a list with arrays
  for j in range(T_d):
    Z_dt = np.kron(np.kron(PsiTilde_td[:,j].T,X_d[j].T),np.eye(n_rx,dtype='complex128'))
    Z_d.append(Z_dt)
    n_dt = np.random.normal(loc=0, scale=np.sqrt(varn/2), size=(n_rx,1*2)).view(np.complex128) #Data noise
    Y_d.append(np.matmul(Z_dt,h[:,np.newaxis]) + n_dt)
  h_initial = np.matmul(np.linalg.pinv(np.vstack(Z_p)),np.vstack(Y_p))
  return Y_p,Y_d,Z_p,Z_d,h_initial


T_d = 50; # Number of data symbols
T_p = [4]; # Number of pilot symbols
# T_p = [4]
N = 32; # Number of IRS elements
n_rx = 2; # Number of receive antennas
n_tx = 2; # Number of tr ansmit antennas
itera = 5; # Number of EM iterations
monte_iter = 3;  # Number of Monte carlo iterations
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
mse = []
for i in range(monte_iter):
    h = channelMatrix(n_tx,n_rx,N,varh)
    X_d,all_possibleSymbols = symbols(n_tx,M_symbols,T_d)    
    PsiTilde_tp,PsiTilde_td = irsMatrix(T_p[0],T_d,N,beta_min,amp)
    PsiTilde_tp = np.insert(PsiTilde_tp,0,np.ones((1,T_p[0]),dtype='complex128'),axis= 0)
    PsiTilde_td = np.insert(PsiTilde_td,0,np.ones((1,T_d),dtype='complex128'),axis= 0)
    # print(np.matmul(PsiTilde_tp,np.conjugate(PsiTilde_tp).T))
    X_p = pilotSymbols(n_tx,M_symbols,T_p[0])
    # X_d,all_possibleSymbols = symbols(T_p[tp],T_d,n_tx,M_symbols)
    Y_p,Y_d,Z_p,Z_d,h_initial = receivedSignals(T_p[0],T_d,PsiTilde_tp,PsiTilde_td ,n_rx,n_tx,X_d,X_p,h,varn,M_symbols,N)
    theta_hat,logLikelihood = em(Y_d,Y_p,T_d,T_p[0],Z_p,Z_d,PsiTilde_td ,all_possibleSymbols,M_symbols,varn,itera,h_initial,n_tx)
    print("original:",norm(h))
    print("predicted",norm(theta_hat))
    mse.append(logLikelihood)
mse = np.average(np.hstack(mse), axis=1)
plt.plot(range(itera), mse, label = 'Proposed method')
plt.grid(color = 'red', linestyle = '--', linewidth = 0.5)
plt.ylabel('LLF')
plt.xlabel('Iterations')
plt.xticks(range(itera))
# plt.yscale("log")
plt.title(' Proposed method with DFT for pilots')
plt.show()