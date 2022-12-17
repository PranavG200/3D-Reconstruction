from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None
    c1 =0
    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]

        """ YOUR CODE HERE
        """
        x1 = X1[sample_indices]
        x2 = X2[sample_indices]
        E = least_squares_estimation(x1,x2)
        c= 0
        inliers1 = []
        for j in test_indices:
          d1 = (X2[j].T @(E @ X1[j]))**2 /(np.linalg.norm(np.cross(np.array([0,0,1]),(E @ X1[j])))**2)
          d2 = (X1[j].T @(E.T @ X2[j]))**2 /(np.linalg.norm(np.cross(np.array([0,0,1]),(E.T @ X2[j])))**2)

          if (d1 + d2 < 1e-4):
            c = c+1
            inliers1.append(j)
        if c > c1:
          c1 = c
          inliers1 = np.array(inliers1)
          inliers = np.concatenate((sample_indices,inliers1))
          """ END YOUR CODE
          """
          #if inliers.shape[0] > best_num_inliers:
          best_num_inliers = inliers.shape[0]
          best_E = E
          best_inliers = inliers


    return best_E, best_inliers