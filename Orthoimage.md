---
aliases:
  - Orthophoto
  - Orthophotomosaic
  - Orthorectification
  - Orthorectify
  - Orthomosaic
---
An aerial photograph or satellite image that's been ==geometrically corrected== ("orthorectified") to remove distortion from camera tilt and terrain relief.
- Corrected-for distortions are caused by **[[Nadir|Off-Nadir]] viewing angles** and **varying terrain elevations** (using a [[Digital Elevation Model|DEM]]).
	- GCPs aren't strictly needed...  Using the DEM and using the sensor's exact position/orientation/look angle at the moment of capture, you can trace a ray from each pixel back through the sensor geometry to the ground, figure out where it actually hit the earth's surface accounting for elevation, and resample the image to where that point should appear on a map, which is doing what's called *rigorous sensor modeling* or using using an [[Rational Polynomial Coefficient]] (RPC) model, a mathematical approximation of that ra-tracing geometry that comes bundled with satellite imagery.
	- [[Ground Control Point]]s can improve absolute, tying the image to known ground coordinates, correcting any residual offset in the sensor model. Essential if you need sub-meter absolute accuracy.
	- GCPs are not needed for relative accuracy: If you just need the image to be internally consistent (correct shape, no distortion), RPCs + DEM alone are sufficient.
	- Using [[Tie Point]]s (matched features between overlapping images) can substitute for [[Ground Control Point|GCP]] when you need consistency across a mosaic but lack surveyed ground truth.
	- In practice, precision applications (mapping, change detection across sensors) use GCPs; many commercial workflows skip them and rely on RPC-only orthorectification


Unlike an uncorrected aerial photograph, an Orthophoto can be used to measure true distances, because it is an accurate representation of the Earth's surface, having been adjusted for topographic relief, lens distortion, and camera tilt.

Orthophotographs are commonly used as a "map accurate" background image in GIS systems.
- A [[Digital Elevation Model]] (DEM) or topographic map is required to create an orthophoto, as distortions in the image due to the varying distance between the camera/sensor and different points on the ground need to be corrected.

![[Pasted image 20260415105645.png|500]]


![[Pasted image 20260415114628.png]]

![[Pasted image 20260415114636.png]]