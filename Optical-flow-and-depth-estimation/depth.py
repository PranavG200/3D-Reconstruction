import numpy as np
import math

def depth(flow, confidence, ep, K, thres=10):
    """
    params:
        @flow: np.array(h, w, 2)
        @confidence: np.array(h, w, 2)
        @K: np.array(3, 3)
        @ep: np.array(3,) the epipole you found epipole.py note it is uncalibrated and you need to calibrate it in this function!
    return value:
        depth_map: np.array(h, w)
    """
    depth_map = np.zeros_like(confidence)

    """
    STUDENT CODE BEGINS
    """
    K_ = K/K[2,2]
    Kinv = np.linalg.inv(K_)
    fx = K_[0,0]
    fy = K_[1,1]

    u = flow[:,:,0]/fx
    v = flow[:,:,1]/fy

    epO = (Kinv@ep)
    epx = epO[0]
    epy = epO[1]

    for i in range(np.shape(u)[0]):
        for j in range(np.shape(u)[1]):

            c = u[i,j]
            d = v[i,j]
            X = Kinv@(np.array([j,i,1]))

            a = X[0]/X[2] - epx
            b = X[1]/X[2] - epy

            if(confidence[i,j]>thres):
                depth_map[i][j] = np.sqrt((a**2 + b**2) / (c**2 + d**2))
    
    """
    STUDENT CODE ENDS
    """

    truncated_depth_map = np.maximum(depth_map, 0)
    valid_depths = truncated_depth_map[truncated_depth_map > 0]
    # You can change the depth bound for better visualization if your depth is in different scale
    depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)
    # print(f'depth bound: {depth_bound}')

    truncated_depth_map[truncated_depth_map > depth_bound] = 0
    truncated_depth_map = truncated_depth_map / truncated_depth_map.max()
    

    return truncated_depth_map
