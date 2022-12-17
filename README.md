The entire pipeline consists of different parts -
1) Using optical flow to get point correspondences and estimate depths.
2) Reconstruction of 3d scene from 2 views using 2 view sfm

We first identify important features using SIFT -
![image](https://user-images.githubusercontent.com/46398827/208264027-4701e79e-c734-42f9-bd95-485382988e0b.png)

We then match key points using both least square and RANSAC to prove effectiveness of ransac -
![image](https://user-images.githubusercontent.com/46398827/208264125-c77cb446-42cf-4060-bd80-c5e7556b631d.png)
![image](https://user-images.githubusercontent.com/46398827/208264129-6685c5ca-1115-4d18-a404-3241050edfbe.png)


Lastly we recreate the 3D model from multi view sfm
