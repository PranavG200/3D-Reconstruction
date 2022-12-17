from weakref import ref
import numpy as np
import cv2
from scipy.fft import dst
from tqdm import tqdm


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """

    points = np.array((
        (0, 0, 1),
        (width, 0, 1),
        (0, height, 1),
        (width, height, 1),
    ), dtype=np.float32).reshape(2, 2, 3)
    points = points.reshape(4, 3).T


    """ YOUR CODE HERE
    """
    R = Rt[:3, :3]
    T = Rt[:3, -1].reshape((-1, 1))
    CamPoints =  ((np.linalg.inv(K))  @ points)
    CamPoints = CamPoints / CamPoints[-1, :]
    CamPoints = depth * CamPoints
    points = ((np.linalg.inv(R) @ (CamPoints - T)).T).reshape(2,2,3)

    """ END YOUR CODE
    """
    return points

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    h,w = points.shape[:2]
    points = points.reshape(-1,3)

    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = points.T
    points = K @ Rt @ points
    points = (points[:2] / points[2]).T
    points = points.reshape(h,w,2) 

    """ END YOUR CODE
    """
    return points

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """
    height, width = neighbor_rgb.shape[:2]

    """ YOUR CODE HERE
    """

    points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
    # points = points.T
    sPoints = backproject_fn(K_ref, width, height, depth, Rt_ref)
    dPoints = project_fn(K_neighbor, Rt_neighbor, sPoints)
    dPoints = dPoints.reshape((dPoints.shape[0]*dPoints.shape[1], 2))
    H, m_ = cv2.findHomography(points, dPoints, cv2.RANSAC)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, np.linalg.inv(H), dsize = (width, height))

    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """

    srcM = np.mean(src, axis = 2)
    dstM = np.mean(dst, axis = 2)
    srcM = srcM[:, :, np.newaxis, :]
    dstM = dstM[:, :, np.newaxis, :]
    zncc = np.sum((src - srcM) * (dst - dstM), axis = 2)/(np.std(src, axis = 2) * np.std(dst, axis = 2) + EPS)
    zncc = np.sum(zncc, axis = 2)

    """ END YOUR CODE
    """

    return zncc 


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    u, v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERE
    """
    xcam = (u - K[0, -1]) * dep_map/K[0, 0]
    ycam = (v - K[1, -1]) * dep_map/K[1, 1]

    xyz_cam = np.dstack((xcam, ycam, dep_map))
    """ END YOUR CODE
    """
    return xyz_cam

