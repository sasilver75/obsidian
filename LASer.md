---
aliases:
  - LAS
---
The standard file format for storing [[Light Detection and Ranging|LiDAR]] point cloud data.
- [[LAZ]] is the compressed version, which is always used when the data is stored.
- The cloud-native successor is [[Cloud-Optimized Point Cloud]] (COPC), which is [[LAZ]] organized into a spatial octree so you can use [[HTTP Range Request]]s to just retrieve the points covering a specific area/resolution without downloading the whole file. It's the same data as LAZ, just different internal organization.
- - *"[[LASer|LAS]] is the point cloud version of a [[Comma-Separated Values|CSV]]; every row is a point with coordinates and attributes. [[LAZ]] is that CSV gzipped, and [[Cloud-Optimized Point Cloud|COPC]] is that CSV spatially indexed for cloud access"* 

LAS was developed and maintained by [[American Society for Photogrammetry and Remote Sensing|ASPRS]].
A LAS file stores an array, each with:
- X/Y/Z coordinates
- Intensity: Strength of return signal
- Return number: First, second, third return (a laser pulse can bounce/penetrate multiple surfaces)
- Number of returns: Total returns from that pulse
- Classification: Ground, vegetation, building ,water, noise
- RGB color (if collected with camera fusion)
- GPS timestamp
- Scan angle


![[Pasted image 20260425222440.png]]