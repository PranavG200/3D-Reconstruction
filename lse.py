import numpy as np

def least_squares_estimation(X1, X2):
  """ YOUR CODE HERE
  """
  f = 552
  u0 = 307.5
  v0 = 205
  n = np.shape(X1)[0]
  K = np.array([[f,0,u0],[0,f,v0],[0,0,1]])

  a= np.zeros((n,9))
  for i in range(n):
      p = X1[i,:]#np.linalg.inv(K)@ 
      q = X2[i,:]#np.linalg.inv(K)@ 
      a[i,0:3] =p[0]*q.T
      a[i,3:6] =p[1]*q.T
      a[i,6:9] =p[2]*q.T
      
  [U, S , Vt ] = np.linalg.svd(a)
  E_ = np.transpose(Vt)[:,-1]
  E1 = np.zeros((3,3))
  E1[:,0] = E_[0:3]
  E1[:,1] = E_[3:6]
  E1[:,2] = E_[6:9]
  [u, s , v] = np.linalg.svd(E1)
  E = u @ (np.diag([1,1,0]) @ v)

  """ END YOUR CODE
  """
  return E
