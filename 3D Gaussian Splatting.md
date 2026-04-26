---
aliases:
  - 3DGS
---
Gaussian Splatting (3DGS) is a technique from 2023 that took the computer vision/graphics world by storm.

It's a method for ==novel view synthesis==: Given a set of photographs from a scene from known camera positions, reconstruct the scene so that you can render it from any new viewpoint in real time.

This same problem was previously dominated by [[Neural Radiance Fields]] (NeRFs), which use a neural network to represent the scene. 3DGS  solves the same problem but without a neural network, using explicit geometry instead.

==3DGS enables real-time rendering==: NeRFs require a NN forward pass per ray, which is slow; 3DGS projects and composites Gaussians on the GPU, which is .. what GPUs are designed for! You get interactive-level framerates!
- Explicit representation: Gaussians are actual geometry you can inspect and manipulate, unlike a black-box neural network.
- High quality: Comparable or better visual quality to NeRF in many scenes
- Fast training: Minutes to hours, rather than days.

# How it works
- The scene is represented as a collection of 3D Gaussian ellipsoids, millions of semi-transparent colored blobs in 3D space.
- Each blob has:
	- Position (XYZ center)
	- Covariance (shape and orientation; how stretched, which direction)
	- Color 
	- Opacity
- To render a new view:
	- Project all Gaussians onto the image plan (they become 2D Gaussian splats, hence the name)
	- Sort by depth (back to front)
	- Alpha-composite them in order

The result is a differentiable renderer: You can optimize the Gaussian parameters using [[Gradient Descent]] against your input photographs.
- Start with a point cloud from Structure from Motion (SfM)
- Initialize Gaussians at each point
- Iteratively adjust their position, shape, color, and opacity until the rendered views match the input photos

# Relevance to Geospatial
- Starting to appear in geospatial contexts, in terms of Urban scene reconstruction, where you build 3DGS models of cities from drone imagery or street-level photos.
- Cultural heritage: Capturing buildings, archaeological site with photorealistic 3D representations.
- Comparison with [[Photogrammetry]]: 3DGS often produces better appearance than traditional mesh photogrammetry, though meshes are more useful for measurement.

==3DGS works beautifully for bounded scenes (building, neighborhood), but doesn't yet scale to city or regional extent the way LiDAR point clouds or photogrammetric DEMs do. Active area of research.==

![[Pasted image 20260425210438.png]]
Above:
- [[COLMAP]] is an open-source, general purpose [[Photogrammetry|Structure from Motion]] (SfM) and [[Photogrammetry|Multi-View Stereo]] (MVS) pipeline used to generate 3D reconstructions and camera poses from ordered or unordered image sets.
	- Offers a GUI and command line interface for tasks like sparse/dense reconstruction, feature matching, and automatic reconstruction.


