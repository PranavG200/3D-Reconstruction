The entire pipeline consists of different parts -
1) Using optical flow to get point correspondences and estimate depths.
2) Reconstruction of 3d scene from 2 views using 2 view sfm

We first identify important features using SIFT -
![Alt text](3D reconstruction from 2D images/Results/SIFT points.png)

We then match key points using both least square and RANSAC to prove effectiveness of ransac -
![image](3D reconstruction from 2D images/Results/Key pts using lst sq.png)
![image](3D reconstruction from 2D images/Results/Key pts using RANSAC.png)

The resulting epipolar lines are as follows 
 ![image](3D reconstruction from 2D images/Results/Epipolar lines.png)

Finally we reproject the points of one image onto the other
![image](3D reconstruction from 2D images/Results/Reprojection.png)

Lastly we recreate the 3D model from multi view sfm
