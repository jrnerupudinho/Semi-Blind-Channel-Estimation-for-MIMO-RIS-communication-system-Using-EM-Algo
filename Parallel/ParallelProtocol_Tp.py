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
  return h[:,np.newaxis]

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

def dataPilotSymbols(n_tx,X_p,X_d):
    T_p = len(X_p)
    T_d = len(X_d)
    X_p =  np.concatenate(X_p,axis = 1)
    X_d = np.concatenate(X_d,axis = 1)
    T = max(T_p,T_d)
    X = np.zeros((n_tx,T),dtype='complex128')
    if T_p>T_d:
        X_d = np.append(X_d,np.zeros((n_tx,T_p-T_d),dtype='complex128'),axis= 1)
    if T_d>T_p:
        X_p = np.append(X_p,np.zeros((n_tx,T_d-T_p),dtype='complex128'),axis= 1)
    X = X_p+X_d
    return X
 
def helper(X_d,X_p,T_p,T_d,n_tx):
     X_p =  np.concatenate(X_p,axis = 1)
     X_d = np.concatenate(X_d,axis = 1)
     if T_p>T_d:
         X_d = np.append(X_d,np.zeros((n_tx,T_p-T_d),dtype='complex128'),axis= 1)
     if T_d>T_p:
         X_p = np.append(X_p,np.zeros((n_tx,T_d-T_p),dtype='complex128'),axis= 1)
     return X_p,X_d

def em(Y,T,Z,X_d,X_p,T_p,T_d,n_tx,PsiTilde_t,all_possibleSymbols,M,varn,itera,N):
  X_p,_ =helper(X_d,X_p,T_p,T_d,n_tx)  
  n_rx,_ = Y[0].shape
  theta_t = np.zeros(((n_tx*n_rx*(N+1)),1),dtype='complex128')
  print("inital theta",norm(theta_t))
  for l in range(itera):
    m_secondTermNumer = np.zeros(((N+1)*n_tx*n_rx,1),dtype='complex128')
    m_secondTermDenom = np.zeros(((N+1)*n_tx*n_rx,(N+1)*n_tx*n_rx),dtype='complex128')
    # beta_denominator = 0 
    for t in range(0,T):
      beta_denominator = 0  
      for k in range(0,np.power(M,n_tx)):
        first_dummy_den = Y[t]-np.matmul(np.kron(np.kron(PsiTilde_t[:,t][np.newaxis],(all_possibleSymbols[k][np.newaxis]+X_p[:,t])),np.eye(n_rx,dtype='complex128')),theta_t)
        beta_denominator += mp.exp(-np.power(norm(first_dummy_den),2)/np.power(varn,2))
      for j in range(0,np.power(M,n_tx)):
        first_dummy = Y[t]-np.matmul(np.kron(np.kron(PsiTilde_t[:,t][np.newaxis],all_possibleSymbols[j][np.newaxis]+X_p[:,t]),np.eye(n_rx,dtype='complex128')),theta_t)
        beta_exp = mp.exp(-np.power(norm(first_dummy),2)/(varn**2))/beta_denominator
        m_secondTermNumer += float(beta_exp)*np.matmul(np.conjugate(np.kron(np.kron(PsiTilde_t[:,t][np.newaxis],all_possibleSymbols[j][np.newaxis]+X_p[:,t]),np.eye(n_rx,dtype='complex128'))).T,Y[t])
        m_secondTermDenom += float(beta_exp)*np.matmul(np.conjugate(np.kron(np.kron(PsiTilde_t[:,t][np.newaxis],all_possibleSymbols[j][np.newaxis]+X_p[:,t]),np.eye(n_rx,dtype='complex128'))).T,(np.kron(np.kron(PsiTilde_t[:,t][np.newaxis],all_possibleSymbols[j][np.newaxis]+X_p[:,t]),np.eye(n_rx,dtype='complex128'))))
    theta_tPlusOne = np.linalg.solve( m_secondTermDenom,  m_secondTermNumer)
    theta_t = theta_tPlusOne
    print(norm(theta_t))
  return theta_t


def irsMatrix(T,N):
  PsiTilde_t = np.zeros((N+1,T),dtype = 'complex128')
  for n in range(0,N+1):
    for t in range(0,T):
            PsiTilde_t[n,t] = np.exp((-1j*2*np.pi*(t)*(n))/(T))
  return PsiTilde_t
    
def receivedSignals(T,PsiTilde_t ,n_rx,n_tx,X,h,varn):
  Z = []
  Y = []
  for j in range(T):
    Z_t = np.kron(np.kron(PsiTilde_t[:,j].T,X[:,j].T),np.eye(n_rx,dtype='complex128'))
    Z.append(Z_t)
    n_t = np.random.normal(loc=0, scale=np.sqrt(varn/2), size=(n_rx,1*2)).view(np.complex128) #Data noise
    Y.append(np.matmul(Z_t,h) + n_t)
  return Y,Z

def main():
    T_d = 50; # Number of data symbols
    T_p = [4,8,12,16,20,24,28,32,36,40]; # Number of pilot symbols
    # T_p = [6]
    N = 32; # Number of IRS elements
    n_rx = 8; # Number of receive antennas
    n_tx = 1; # Number of tr ansmit antennas
    itera = 20; # Number of EM iterations
    monte_iter = 50;  # Number of Monte carlo iterations
    varh = 1; # The variance of the channels
    M_symbols = 16; # Constellation size for data and pilot symbols.
    power = 10 #constellation power, needs to be computed from the constellation. 
    SNR = [20] #SNR (dB)
    varn = 0.1
    
    
    # #Computation 
    mse = np.zeros((monte_iter,len(T_p)))
    for i in range(monte_iter):
        h = channelMatrix(n_tx,n_rx,N,varh)
        X_d,all_possibleSymbols = symbols(n_tx,M_symbols,T_d)
        for tp in range(len(T_p)):
          T = max(T_d,T_p[tp])
          PsiTilde_t = irsMatrix(T,N)
          X_p = pilotSymbols(n_tx,M_symbols,T_p[tp])
          X = dataPilotSymbols(n_tx,X_p,X_d)
          Y,Z = receivedSignals(T,PsiTilde_t ,n_rx,n_tx,X,h,varn)
          theta_hat = em(Y,T,Z,X_d,X_p,T_p[tp],T_d,n_tx,PsiTilde_t,all_possibleSymbols,M_symbols,varn,itera,N)
          print("original:",norm(h))
          print("predicted",norm(theta_hat))
          mse[i][tp] = np.matrix.trace(np.abs(np.matmul((np.conjugate(theta_hat-h).T),(theta_hat-h))))/np.power(norm(h),2)
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
    return

main()


