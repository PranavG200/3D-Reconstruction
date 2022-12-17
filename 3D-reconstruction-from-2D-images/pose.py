import numpy as np

def pose_candidates_from_E(E):
  transform_candidates = []
  ##Note: each candidate in the above list should be a dictionary with keys "T", "R"
  """ YOUR CODE HERE
  """
  [U, S , Vt ] = np.linalg.svd(E)
  [U1, S1 , Vt1 ] = np.linalg.svd(-E)

  Rzp = np.array([[0,-1,0],[1,0,0],[0,0,1]])
  Rzn = np.array([[0,1,0],[-1,0,0],[0,0,1]])

  transform_candidates.append({"T": U[:,2],"R": U @(Rzp.T @ Vt)})
  transform_candidates.append({"T": U[:,2],"R": U @(Rzn.T @ Vt)})
  transform_candidates.append({"T": -U[:,2],"R": U @(Rzp.T @ Vt)})
  transform_candidates.append({"T": -U[:,2],"R": U @(Rzn.T @ Vt)})#/np.linalg.norm(U[:,2]))
  
  """ END YOUR CODE
  """
  return transform_candidates