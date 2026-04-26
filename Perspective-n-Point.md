---
aliases:
  - PnP
---
The problem of recovering a camera's ==pose== (position and orientation)... GIVEN:
1. A set of **n 3D points** in world coordinates (already known)
2. Their corresponding **2D projections** in the image (pixel coordinates)
3. the camera's **intrinsics**


Intuition:
- You already know where these 3D points are in the world, and you know where they appear in the 2D image, so you can work backwards to figure out where the camera must have been an which way it was pointing to produce that image.

In [[COLMAP]]'s pipeline, at each step of incremental reconstruction:
1. You have a growing set of triangulated 3D points.
2. A new image comes in, [[Scale-Invariant Feature Transforms|SIFT]] matches its keypoints to those known 3D points.
3. [[Perspective-n-Point|PnP]] solves for the image's camera pose, given those 2D <-> 3D correspondences.
4. That poses gets added to the reconstruction, more points get triangulated, repeat.

Minimum cases:
- P3P: Minimum 3 points need for a finite set of solutions (up to 4)
- P4P+: More points overdetermine the system, used with RANSAC to handle outliers.