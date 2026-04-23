---
aliases:
  - VFM
---
Processes images only, pretrained to learn general visual representations, then finetuned for visual tasks (segmentation, detection, classification).
- [[SatMAE]], [[SkySense]], [[Segment Anything Model|SAM]]

In the [[Remote Sensing]] domain, VFMs are expected ot handle a diverse array of visual sensor modalities beyond RGB images, like optical [[Multispectral|MSI]], [[Hyperspectral|HSI]], [[Synthetic Aperture Radar|SAR]] images, [[Thermal Infrared]] images, and 3D [[Light Detection and Ranging|LiDAR]] point clouds.

Two main approaches:
- Pre-training models (either supervised or self-supervised)
- [[Segment Anything Model|SAM]]-based models


Compare with a different thing, a [[VLM|Vision-Language Model]] (Examples: [[GeoChat]], [[EarthGPT]], [[GRAFT]])
- VLMs can answer "How many ships are in this image" and "Why did you think that?" (unclear if it knows it's not lying)
