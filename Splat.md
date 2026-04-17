In point cloud visualization (e.g. [[Potree]] visualization of [[Light Detection and Ranging|LiDAR]] data), a "splat" is simply ==how you render a point that has no inherent size.==
- A LiDAR point is mathematically dimensionless, it's just an XYZ coordinate.
- If you render it as a single pixel, it looks sparse and hard to read, especially when zoomed out.
- ==A splat gives each point a small screen-space disc or circle, sized to appropriately fill the gap between neighboring points so that the result looks like a continuous surface rather than a sparse constellation of dots.==
	- Disc size is typically computed from the local point spacing and camera distance.


See: [[Gaussian Splatting]]