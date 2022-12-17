The entire pipeline consists of different parts -
1) Using optical flow to get point correspondences and estimate depths.

Optical flow is computed first. The smallest singular value of spatiotemporal derivative matrix is calculated and only those pixels which are above a threshold value are considered. the flow vectors are shown below 
![Alt text](Optical-flow-and-depth-estimation/Results/flow_10.png)

Epipoles after RANSAC and satisfying planar condition equation by a threshold
![Alt text](Optical-flow-and-depth-estimation/Results/epipole_10.png)

Depths are then calculated by assuming pure translational motion
![Alt text](Optical-flow-and-depth-estimation/Results/depth_10.png)

2) Reconstruction of 3d scene from 2 views using 2 view sfm

We first identify important features using SIFT -
![Alt text](3D-reconstruction-from-2D-images/Results/SIFT-points.png)

We then match key points using both least square and RANSAC to prove effectiveness of ransac -
![Alt text](3D-reconstruction-from-2D-images/Results/Key-pts-using-lst-sq.png)
![Alt text](3D-reconstruction-from-2D-images/Results/Key-pts-using-RANSAC.png)

The resulting epipolar lines are as follows 
![Alt text](3D-reconstruction-from-2D-images/Results/Epipolar-lines.png)

Finally we reproject the points of one image onto the other
![Alt text](3D-reconstruction-from-2D-images/Results/Reprojection.png)

3) Lastly we recreate the 3D model from multi view sfm

Input views - 
![Alt text](Reconstruction-from-Multi-view-stereo/Results/Input-views.png)

Disparity -
![Alt text](Reconstruction-from-Multi-view-stereo/Results/Disparity.png)

Disparity and depth after post processing -
![Alt text](Reconstruction-from-Multi-view-stereo/Results/Postproc-Disparity-and-depth.png)

L-R Consistency check mask -
![Alt text](Reconstruction-from-Multi-view-stereo/Results/L-R-Consistency-Check-Mask.png)

Reconstructed 3d model from 2 views using ZNCC Kernel -
![Alt text](Reconstruction-from-Multi-view-stereo/Results/Reconstructed-3d-model-ZNCC.png)

Entire Reconstructed 3d model
![Alt text](Reconstruction-from-Multi-view-stereo/Results/Reconstructed-3d-model.png)
