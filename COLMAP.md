 [[COLMAP]] is an open-source, general purpose [[Photogrammetry|Structure from Motion]] (SfM) and [[Photogrammetry|Multi-View Stereo]] (MVS) pipeline used to generate 3D reconstructions and camera poses from ordered or unordered image sets.
- Offers a GUI and command line interface for tasks like sparse/dense reconstruction, feature matching, and automatic reconstruction.
- ==The standard preprocessing step for both [[Neural Radiance Fields|NeRF]]s and [[3D Gaussian Splatting|3DGS]]==; Both need camera poses as input, and COLMPA is how you get them from raw photos.

What it produces, given N photos of a building/object/scene:
1. ==Camera poses==: Where each camera was, in 3D space, and its orientation.
2. ==Sparse point cloud==: 3D point clouds reconstructed from matched features across images.
3. ==Dense point cloud== ([[Photogrammetry|MVS]] stage): A much denser 3D reconstruction.
4. ==Camera intrinsics==: Focal length, distortion, etc. if not known.

How it works:
- [[Photogrammetry|SfM]] stage:
	- Feature extraction: detect keypoints in every image ([[Scale-Invariant Feature Transforms|SIFT]] by default)
	- Feature matching: find the same keypoints across different images
	- Incremental reconstruction: start with an image pair, triangulate 3D points, and ad more images one by one, solving for their pose, bundle adjust throughput to minimize reprojection error.
- [[Photogrammetry|MVS]] stage:
	- Uses the recovered camera poses to do dense stereo matching across many image pairs, producing a dense point cloud or depth maps.

## Relationship to [[Coregistration]]
- Related but different. Coregistration aligns 2D images to eachother, while COLMAP recovers the 3D geometry and camera positions that explain why images look the way they do.

