---
aliases:
  - RANSAC
---
An iterative algorithm for fitting a model to data that contains outliers, by repeatedly trying random minimal subset and seeing which model gets the most ==support==.
- Support: The count of points that are consistent with a candidate model, i.e. the "inlier count." A point "supports" a model if its error is below some threshold. The model with the most supporting points wins.

Core Loop, for many iterations:
1. Sample the minimum number of points needed to fit the model
2. Fit the model to only those points
3. Count how many OTHER points agree with this model (==inliers==)

Return the model with the most inliers
Then do a final fit using all inliers of the best model.

You have 100 points, 30 of which are outliers. [[Least-Squares]] would get pulled toward the outliers. RANSAC:
  1. Pick 2 random points, fit a line
  2. Count how many of the 100 points are within ε distance of that line
  3. Repeat 100s of times
  4. The true line will eventually be sampled, get ~70 inliers, and win

In COLMAP/PnP:
- Outliers are wrong feature matchers: SIFT matched a keypoint to the wrong 3D point.
- RANSAC:
	- Sample 3 2D<->3D correspondences
	- Solve [[Perspective-n-Point|PnP]] -> Candidate camera pose
	- Count how many other correspondences agre with taht pose (reproject 3D point, check distance to 2D keypoint)
	- Keep bset pose

The key parameter is the number of iterations, chosen based on expected outlier ratio. More outliers = more iterations needed.