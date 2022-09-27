#The comparision code for both proposed and approx method for T_p vs MSE. 
#16th sept - the code is still unfunctional.

import numpy as np
import QAM as qp
from numpy.linalg import norm
import matplotlib.pyplot as plt
import itertools
import mpmath as mp
from scipy import linalg 
def channelMatrix1(varh,N,M,n_r,n_rx,n_tx):
  h = np.random.normal(loc=0, scale=np.sqrt(varh/2), size=(N, n_r*2)).view(np.complex128) #H_BS #*2 because we need two values to create a complex number view
  F_H = np.random.normal(loc=0, scale=np.sqrt(varh/2), size=(M, N*2)).view(np.complex128)
  H_SU = F_H
  H_BS = h
  G = np.matmul(np.diagflat(np.conjugate(h).T),np.conjugate(F_H).T)
  h_proposed = linalg.khatri_rao(H_BS.T, H_SU).flatten()
  # print(np.power(norm(G),2))
  return h,F_H, G,h_proposed


def symbols(n_tx,M,T_d):
  # X_p = []
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

def em_proposed(Y_d,Y_p,T_d,T_p,Z_p,PsiTilde_td ,all_possibleSymbols,M,varn,itera,h_initial):
  
  n_rx,_ = Y_d[0].shape
  theta_t = h_initial 

  for l in range(itera):
    m_secondTermNumer = 0
    m_secondTermDenom = 0
    m_firstTermNumer = 0
    m_firstTermDenom = 0
    for t in range(0,T_d):
      beta_denominator = 0
      for k in range(0,np.power(M,n_tx)):
        first_dummy_den = Y_d[t]-np.matmul((np.kron(np.kron(PsiTilde_td[:,t][np.newaxis],all_possibleSymbols[k][np.newaxis]),np.eye(n_rx,dtype='complex128'))),theta_t)
        beta_denominator += mp.exp(-np.power(norm(first_dummy_den),2)/np.power(varn,2))
      for j in range(0,np.power(M,n_tx)):
        first_dummy = Y_d[t]-np.matmul((np.kron(np.kron(PsiTilde_td[:,t][np.newaxis],all_possibleSymbols[j][np.newaxis]),np.eye(n_rx,dtype='complex128'))),theta_t)
        beta_exp = mp.exp(-np.power(norm(first_dummy),2)/(varn**2))/beta_denominator
        m_secondTermNumer += float(beta_exp)*np.matmul(np.conjugate(np.kron(np.kron(PsiTilde_td[:,t][np.newaxis],all_possibleSymbols[j][np.newaxis]),np.eye(n_rx,dtype='complex128'))).T,Y_d[t])
        m_secondTermDenom += float(beta_exp)*np.matmul(np.conjugate(np.kron(np.kron(PsiTilde_td[:,t][np.newaxis],all_possibleSymbols[j][np.newaxis]),np.eye(n_rx,dtype='complex128'))).T,(np.kron(np.kron(PsiTilde_td[:,t][np.newaxis],all_possibleSymbols[j][np.newaxis]),np.eye(n_rx))))
    for t in range(0,T_p):
      m_firstTermNumer += np.matmul(np.conjugate(Z_p[t]).T,Y_p[t])
      m_firstTermDenom += np.matmul(np.conjugate(Z_p[t]).T,Z_p[t])
    theta_tPlusOne = np.linalg.solve((m_firstTermDenom + m_secondTermDenom),(m_firstTermNumer + m_secondTermNumer))
    theta_t = theta_tPlusOne
  return theta_t

def EM_approx(PsiTilde_tp,PsiTilde_td,x_p,Y_p,Y_d,Iterations,varn,G):  
  S_p = np.matmul(PsiTilde_tp,np.diagflat(x_p))
  G_cap = np.conjugate(np.matmul(Y_p, np.linalg.pinv(S_p))).T
  M,L_p  = Y_p.shape
  M,L_d = Y_d.shape
  N,_ = PsiTilde_td.shape
  i = 0
  print("original G", norm(G))
  while i <= Iterations:
    g_T = []
    first_term_inv = np.zeros((N,N),dtype='complex128')
    second_term_inv = np.zeros((N,N),dtype='complex128')
    conditional_mean = np.zeros((1,L_d),dtype='complex128')
    for t1 in range(L_p):
        first_term_inv += np.power(np.abs(x_p[t1]),2)*np.matmul(PsiTilde_tp[:,t1][:,np.newaxis],np.conjugate(PsiTilde_tp[:,t1][:,np.newaxis]).T)
    for t in range(L_d):
        psi_G = np.matmul(np.conjugate(PsiTilde_td[:,t][:,np.newaxis]).T,G_cap) #vector 
        dummy_first = np.matmul(np.matmul(np.conjugate(G_cap).T,PsiTilde_td[:,t][:,np.newaxis]),np.matmul(np.conjugate(PsiTilde_td[:,t][:,np.newaxis]).T,G_cap))
        inver = np.linalg.inv(dummy_first + np.power(varn,2)*np.identity(M))
        dummySecond = np.matmul(psi_G,np.matmul(inver,Y_d[:,t][:,np.newaxis]))
        conditional_mean[:,t] = dummySecond
        #conditional variance
        product = np.matmul(np.matmul(PsiTilde_td[:,t][:,np.newaxis].T,G_cap),np.matmul(np.conjugate(G_cap).T,PsiTilde_td[:,t][:,np.newaxis])) + np.power(varn,2)
        var_first = np.power(varn,2)*np.linalg.inv(product)
        var_second = np.matmul(np.conjugate(dummySecond).T,dummySecond)                                                  
        conditional_var = var_first + var_second
        second_term_inv += np.matmul(np.matmul(PsiTilde_td[:,t][:,np.newaxis],conditional_var),np.conjugate(PsiTilde_td[:,t][:,np.newaxis]).T)
    for m in range(M):
        first_term = np.zeros((1,N),dtype='complex128')
        second_term = np.zeros((1,N),dtype='complex128')
        for t1 in range(L_p):
            first_term += Y_p[m,t1]*np.conjugate(x_p[t1])*np.conjugate(PsiTilde_tp[:,t1][:,np.newaxis]).T
        for t in range(L_d):
            second_term += Y_d[m,t]*np.conjugate(conditional_mean[:,t])*np.conjugate(PsiTilde_td[:,t][:,np.newaxis]).T
        g_T.append(np.conjugate(np.matmul((first_term + second_term),np.linalg.inv(first_term_inv + second_term_inv))))
    G_cap_iter = np.row_stack(g_T).T
    i =i+1 
    G_cap = G_cap_iter
  print("G_Cap",norm(G_cap))
  return G_cap

def irsMatrix(T_p,T_d,N,beta_min,amp):
  PsiTilde_tp = np.zeros((N,T_p),dtype = 'complex128')
  # PsiTilde_td = np.zeros((N,T_d),dtype = 'complex128')
  
  PsiTilde_td =  []
  for n in range(0,N):
    for t in range(0,T_p):
            PsiTilde_tp[n,t] = np.exp((-1j*2*np.pi*(t)*(n))/(N))
  # for n in range(0,N):
  #    for t in range(0,T_d):
  #            PsiTilde_td[n,t] = np.exp((-1j*2*np.pi*(t)*(n))/(T_d))
  for t in range(0,T_d):
      beta = (beta_max-beta_min)*np.random.uniform(0,1,(N,1))+beta_min; 
      psi_t = amp*np.exp(1j*beta);
      PsiTilde_td.append(psi_t)  
  PsiTilde_td =np.concatenate( PsiTilde_td, axis=1 )
  return PsiTilde_tp, PsiTilde_td
    
def received_proposed(T_p,T_d,PsiTilde_tp,PsiTilde_td ,n_rx,n_tx,X_d,X_p,h,varn,M_symbols):
  Z_p = []
  Z_d = []
  Y_p = []
  Y_d = []
  for i in range(T_p):
    Z_pt = np.kron(np.kron(PsiTilde_tp[:,i].T,X_p[i].T),np.eye(n_rx,dtype='complex128'))
    # print(Z_pt.shape)
    Z_p.append(Z_pt)
    n_pt = np.random.normal(loc=0, scale=np.sqrt(varn/2), size=(n_rx, 1*2)).view(np.complex128) #Pilot noise
    Y_p.append(np.matmul(Z_pt,h)[:,np.newaxis] + n_pt) #creates a list with arrays
  for j in range(T_d):
    Z_dt = np.kron(np.kron(PsiTilde_td[:,j].T,X_d[j].T),np.eye(n_rx,dtype='complex128'))
    Z_d.append(Z_dt)
    n_dt = np.random.normal(loc=0, scale=np.sqrt(varn/2), size=(n_rx,1*2)).view(np.complex128) #Data noise
    Y_d.append(np.matmul(Z_dt,h)[:,np.newaxis] + n_dt)
  h_initial = np.matmul(np.linalg.pinv(Z_p),Y_p)
  return Y_p,Y_d,Z_p,Z_d,h_initial

def noise_approx(varn,M,L_p,L_d):
  z_p = np.random.normal(loc=0, scale=np.sqrt(varn/2), size=(M, L_p*2)).view(np.complex128) #Pilot noise
  z_d = np.random.normal(loc=0, scale=np.sqrt(varn/2), size=(M, L_d*2)).view(np.complex128) #Data noise
  return z_p,z_d

#received signals
def received_approx(h,F_H,PsiTilde_tp,PsiTilde_td,x_d,x_p,z_p,z_d):
  y_p = []
  y_d = []
  _,L_p = z_p.shape
  _,L_d = z_d.shape
  for t in range(0,L_p):
      y_p.append(np.matmul(np.matmul(F_H,np.diagflat(PsiTilde_tp[:,t])),h)*x_p[t]+z_p[:,t][:,np.newaxis])
  for t in range(0,L_d):
      y_d.append(np.matmul(np.matmul(F_H,np.diagflat(PsiTilde_td[:,t])),h)*x_d[t]+z_d[:,t][:,np.newaxis])
  Y_p = np.column_stack(y_p)
  Y_d = np.column_stack(y_d)
  return Y_p,Y_d

T_d = 50; # Number of data symbols
L_d = T_d
T_p = [4,8,12,16,20,24,28,32,36,40]; # Number of pilot symbols
L_p = T_p
N = 32; # Number of IRS elements
n_rx = 8
M = n_rx; # Number of receive antennas
n_tx = 1; # Number of tr ansmit antennas
n_r = n_tx;
itera = 15; # Number of EM iterations
monte_iter = 50;  # Number of Monte carlo iterations
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
MSE_approx = np.zeros((monte_iter,len(L_p)))
MSE_proposed = np.zeros((monte_iter,len(T_p)))
for i in range(monte_iter):
    h,F_H,G,h_proposed = channelMatrix1(varh,N,M,n_r,n_rx,n_tx)
    X_d,all_possibleSymbols = symbols(n_tx,M_symbols,T_d)
    x_d = np.vstack(X_d) #inorder to facilite the difference in the model
    # x_d = dataSymbols_approx(N,M,L_d,n_r,varx)
    # x_d =  dataSymbols_approx(L_d,M_symbols)
    for tp in range(len(T_p)):      
      PsiTilde_tp,PsiTilde_td = irsMatrix(T_p[tp],T_d,N,beta_min,amp)
      # PsiTilde_tp = np.insert(PsiTilde_tp,0,np.ones((1,T_p[tp]),dtype='complex128'),axis= 0)
      # print(np.matmul(PsiTilde_tp,np.conjugate(PsiTilde_tp).T))
      X_p = pilotSymbols(n_tx,M_symbols,T_p[tp])
      x_p = np.vstack(X_p)
       # x_p = pilotsSymbols_approx(N, M, L_p[tp], n_r, varx) 
      Y_p,Y_d,Z_p,Z_d,h_initial = received_proposed(T_p[tp],T_d,PsiTilde_tp,PsiTilde_td ,n_rx,n_tx,X_d,X_p,h_proposed,varn,M_symbols)
      z_p,z_d = noise_approx(varn,M,L_p[tp],L_d)
      Y_p_dft,Y_d_dft = received_approx(h,F_H,PsiTilde_tp,PsiTilde_td,x_d,x_p,z_p,z_d)
      theta_hat = em_proposed(Y_d,Y_p,T_d,T_p[tp],Z_p,PsiTilde_td ,all_possibleSymbols,M_symbols,varn,itera,h_initial)
      G_cap_dft = EM_approx(PsiTilde_tp,PsiTilde_td,x_p,Y_p_dft,Y_d_dft,itera,varn,G)
      print("original:",norm(h_proposed))
      print("predicted",norm(theta_hat))
      MSE_proposed[i][tp] = np.matrix.trace(np.abs(np.matmul((np.conjugate(theta_hat-h_proposed[:,np.newaxis]).T),(theta_hat-h_proposed[:,np.newaxis]))))/np.power(norm(h_proposed[:,np.newaxis]),2)
      MSE_approx[i][tp] = np.matrix.trace(np.abs(np.matmul((np.conjugate(G_cap_dft-G).T),(G_cap_dft-G))))/np.power(norm(G),2)
      # mse[i][tp] = np.power(norm(np.subtract(theta_hat,h[:,np.newaxis])),2)/np.power(norm(h[:,np.newaxis]),2)
    if(i%5 == 0):
      print("The monte carlo Iteration Number is: ",i)
MSE_approx = np.average(MSE_approx, axis=0)
MSE_proposed = np.average(MSE_proposed, axis=0)
plt.plot(T_p, MSE_proposed, label = 'Proposed method')
plt.plot(T_p, MSE_approx, label = 'Approximation method')
plt.grid(color = 'red', linestyle = '--', linewidth = 0.5)
plt.ylabel('NMSE')
plt.xlabel('T_p')
plt.xticks(T_p)
plt.yscale("log")
plt.title(' Comparison between proposed method and the semi-blind Guassian method')
plt.legend()
plt.show()



