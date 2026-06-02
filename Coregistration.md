---
aliases:
  - Coregistered
---


The process of aligning multiple images of the same area so that the same ground point falls on the same pixel across all images.
- Satellites don't image from the same position or angle on every pass; two acquisitions of the same location will have:
	- Slightly different viewing angles
	- Different orbital positions
	- Accumulated geometric distortions
	- Sub-pixel to multi-pixel offsets between them
- If you stack them without coregistration, a pixel at row 100, col 200 in image A and row 100, col 200 in image B might correspond to ground points meters apart. 
	- Without coregistration, any analysis that compares across images (Change Detection, [[Interferometric Synthetic Aperture Radar|InSAR]], time series, compositing, etc.) produces garbage.
- After coregistration, pixel (i,j) in image A and pixel (i,j) in image B point to the same ground point.
- Critical for [[Interferometric Synthetic Aperture Radar|InSAR]], which measures surface deformity by comparing the phase difference of two SAR acquisitions. Phase is extremely sensitive, a shift of half-a-wavelength (~3cm for [[C-Band]]) completely changes the measurement. Sub-pixel misalignment introduces phase errors that swamp the deformation signal you're trying to measure.
	- InSAR coregistration needs to be accurate to ~0.001 pixels! Far more precise than optical change detection, which can tolerate ~0.1-0.3 pixel errors.
- Coregistration and [[Orthoimage|Orthorectification]] are similar but different:
	- ==Orthorectification==: Correct a single image for terrain distortion using a [[Digital Elevation Model|DEM]], removing the displacement caused by hills and valleys. Each image is processed independently.
	- ==Coregistration==: Aligns two or more images to eachother. Can be done after orthorectification, or directly between images.


![[Pasted image 20260425211409.png]]
