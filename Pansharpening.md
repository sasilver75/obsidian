A technique for ==merging a high-resolution [[Panchromatic]] band with a lower-resolution [[Multispectral]] (color) band== to produce a high-resolution color image.

Satellites often have a ==tradeoff== built into their sensor design:
- ==Panchromatic== band: Captures a ==wide range of wavelengths==, so gets ==more light==, so can use a ==smaller pixel== (high [[Resolution|Spatial Resolution]])
- ==Multispectral== band: Captures ==narrow wavelength ranges== (red, green, blue, NIR), so you capture ==less light== per band, and need ==bigger pixels==, leading to lower spatial resolution.

For example, on [[Landsat]] 8:
- Panchromatic: 15m resolution
- Multispectral: 30m resolution

Pansharpening:
- Combines the *spatial detail* from the panchromatic band with the color information from the multispectral band, giving you the best of both:

`High-res pan (grayscale) + Low-res multispectral (color) -> High-res multispectral (color + detail)`

Common algorithms:
- Brovey Transform: Simple ratio-based, fast, but can distort colors
- IHS (Intensity-Hue-Saturation): Replaces intensity channel with pan band
- PCA (Principal Component Analysis): Replace first principal component with pan
- Gram-Schmidt: More sophisticated, better color preservation, used by ESRI/ENVI

Limitations:
  - Always involves some color distortion tradeoff
  - Spectral accuracy suffers somewhat — not ideal if you need precise reflectance
  values for indices like NDVI
  - ==More of a visualization tool than an analysis tool==

  ==Very common in commercial satellite imagery pipelines — most "high-res" imagery you==
  ==see from Maxar or Planet has been pansharpened.==




