import numpy as np
def epipole(u,v,smin,thresh,num_iterations = 10):
    ''' Takes flow (u,v) with confidence smin and finds the epipole using only the points with confidence above the threshold thresh 
        (for both sampling and finding inliers)
        params:
            @u: np.array(h,w)
            @v: np.array(h,w)
            @smin: np.array(h,w)
        return value:
            @best_ep: np.array(3,)
            @inliers: np.array(n,) 
        
        u, v and smin are (h,w), thresh is a scalar
        output should be best_ep and inliers, which have shapes, respectively (3,) and (n,) 
    '''

    """
    You can do the thresholding on smin using thresh outside the RANSAC loop here. 
    Make sure to keep some way of going from the indices of the arrays you get below back to the indices of a flattened u/v/smin
    STUDENT CODE BEGINS
    """
    X = np.linspace(-256,255,512)
    xp, yp = np.meshgrid(X, X)
    sminF = smin.flatten()
    OrigIndices = np.where(sminF>thresh)[0]
    num = len(OrigIndices)

    xpT = xp.flatten()[sminF > thresh]
    ypT = yp.flatten()[sminF > thresh]
    Xp = np.vstack((xpT,ypT,np.ones(num)))

    uT = u.flatten()[sminF > thresh]
    vT = v.flatten()[sminF > thresh]
    U = np.vstack((uT, vT, np.zeros(num)))   

    CrossP = np.cross(Xp.T,U.T)

    """ 
    STUDENT CODE ENDS
    """
    sample_size = 2
    eps = 10**-2

    best_num_inliers = -1
    best_inliers = None
    best_ep = None

    for i in range(num_iterations): #Make sure to vectorize your code or it will be slow! Try not to introduce a nested loop inside this one
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(0,np.sum((smin>thresh))))
        sample_indices = permuted_indices[:sample_size] #indices for thresholded arrays you find above
        test_indices = permuted_indices[sample_size:] #indices for thresholded arrays you find above

        """
        STUDENT CODE BEGINS
        """
        CrossPS = CrossP[sample_indices]
        
        up, s, vh = np.linalg.svd(CrossPS)
        ep = np.transpose(vh)[:,-1]

        inliers = OrigIndices[sample_indices]

        CrossPT = CrossP[test_indices]
        dist = np.abs(CrossPT@ep)

        inliers = np.append(inliers, OrigIndices[test_indices[np.where(dist< eps)[0]]])
         
        """
        STUDENT CODE ENDS
        """

        #NOTE: inliers need to be indices in flattened original input (unthresholded), 
        #sample indices need to be before the test indices for the autograder
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_ep = ep
            best_inliers = inliers

    return best_ep, best_inliers