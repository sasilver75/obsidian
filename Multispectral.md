---
aliases:
  - Multispectral Imaging
  - MSI
---


c.f.: [[Hyperspectral]]


Optical Sensors work very similar to traditional RGB sensors, just have more spectral bands.
- Usually ==3-15 disjoint spectral bands==
- Similar to a normal camera
- Choice of spectral bands (design of sensor) is typically traditional to a specific application (e.g. agriculture)
	- Typically visible + infrared + thermal, maybe ever radar wavelengths of light.
- Common examples: [[Landsat]], [[Sentinel|Sentinel-2]], [[Moderate Resolution Imaging Spectroradiometer|MODIS]], [[Maxar|WorldView]]


Most multi-spectral satellite images are composed of various bands (Red, Green, Blue [[Near Infrared|NIR]]), accurately capturing colors.
However, these images have lower spatial resolutions (smaller bands mean less light is collected, so the resolution can't be as tight).

In contrast, satellites like Landsat-8 also capture [[Panchromatic]] images; these are bands that have a single color, but capture more detail. [[Pansharpening]] involves fusing multispectral images (which have lower resolution) with panchromatic images (which have no color but higher resolution). 

![[Pasted image 20260420194445.png]]

![[Pasted image 20260422113849.png]]

![[Pasted image 20260422114126.png]]
- The portion of the EM spectrum that these satellites cpature is very different, even looking at different multispectral ones.
- Looking at L7 (Landsat-7), there are many different spectral bands
	- 1,2,3, are roughly R,G,B
	- You have some infrared light (4) and longwave infrared (5), and then you have everything down in the thermal bands as well.
- See that Landsat 7 seems to add a few new bands (1), (9)
- Each of these bands, each of these image channels, will capture very different information, and even have different resolutions.
- See band 8? This is a special [[Panchromatic]] band, which spans a wide swatch of the EM spectrum, including most of the visible light, but you get very high resolutions.


![[Pasted image 20260416212902.png]]


