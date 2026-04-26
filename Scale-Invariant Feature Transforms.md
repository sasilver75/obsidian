---
aliases:
  - SIFT
---
An algorithm for detecting and describing **local features** in images that are invariant to scale, rotation, and partially-invariant to illumination and viewpoint changes.

Two steps:
1. ==Keypoint detection== (finding interesting points)
	1. Uses a [[Difference of Gaussians]] (DoG) across multiple scales; blurs the image at increasing scales and looks for points that are local extrema across both space and scale. These tend to be corners, blobs, edges: structurally distinctive points.
	2. The scale-space approach means it finds the same point whether the object is near or far in the image.
2. ==Descriptor computation== (describe each point)
	1. For each keypoint:
		- Take a 16x16 pixel patch around it
		- Divide into 4x4 cells
		- Compute gradient orientation histograms in each cell (8 bins each)
		- Concatenate into a 128-dimensional vector
	- This 128-d vector is the ==SIFT descriptor==, a compact fingerprint of the local appearance around that point
	- Why it's robust:
		- Scale invariant: Detected at the right scale in the pyramid, descriptor normalized to that scale.
		- Rotation invariant: Descriptor is rotated to align with the dominant gradient orientation.
		- Partially illumination invariant: Uses gradients (relative changes), not raw intensities

Matchin