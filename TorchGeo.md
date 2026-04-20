[[PyTorch]]'s official geospatial deep learning library from [[Microsoft Research|MSR]], analogous to torchvision but built for [[Remote Sensing]] and [[Remote Sensing|Earth Observation]] tasks.

Raw geospatial data is hostile to standard ML pipelines:
- Imagery has arbitrary [[Coordinate Reference System]]s, resolution, and extent: You can't just stack tensors.
- Multi-sensor fusion requires spatial alignment across datasets.
- [[Remote Sensing|EO]] datasets have non-standard splits (spatial autocorrelation breaks random train/test splits)

Provides:
- Datasets
	- 50+ ready-to-use EO datasets ([[Sentinel]], [[Landsat]], NAIP, BigEarthNet, etc.)
	- Handles download, checksum, CRS normalization automatically.
- Samplers
	- Geospatial-aware sampling: Random geographic patches, tiles by grid, etc.
	- Respects spatial boundaries to avoid data leakage between train/test splits.
- Transforms:
	- Handles arbitrary band counts (not just RGB), e.g. 13-band Sentinel-2
	- Spectral index computation ([[Normalized Difference Vegetation Index|NDVI]], etc.)
- Pretrained Models:
	- Hosts pretrained weights for EO models like [[Clay]] and other foundation models.


![[Pasted image 20260420104012.png]]
From: https://www.youtube.com/watch?v=HPlZbkNa6zI 
![[Pasted image 20260420104127.png]]
If you don't do overlapping patches you get strange edge/boundary effects, typically does worse at the edge.
These things (see the video above)... are what separates the kids from the adults.