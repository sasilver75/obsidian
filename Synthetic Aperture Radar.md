---
aliases:
  - SAR
---
Resources:
- [NASA EarthData: Synthetic Aperture Radar](https://www.earthdata.nasa.gov/learn/earth-observation-data-basics/sar) (Unread, looks great)

A [[Remote Sensing]] technique that uses radar signals instead of visible light to image the Earth's surface.

A satellite emits microwave pulses towards the ground and records the energy that bounces back; as the satellite moves forwards along its orbit, it takes many measurements from slightly different positions.

The *synthetic aperture* part refers to the fact that you mathematically combine these measurements, as if we had a single enormous antenna that stretched along the flight path, to produce a much sharper image than a real antenna of that size could achieve.
- As the satellite moves along its orbit, it repeatedly emits and receives pulses at the same target from slightly different positions; the onboard computer combines all of these observations mathematically, as if they can from a single enormous "synthesized" antenna.

Why it matters:
- ==Sees through clouds and darkness==: Radar doesn't depend on sunlight, and penetrates cloud cover.
	- Optical satellites like [[Sentinel|Sentinel-2]] and [[Landsat]] are useless over persistent cloud cover, while SAR still has value.
- ==Detects surface roughness and moisture==: The backscatter signal encodes how rough, wet, or metallic a surface is, not just its color. A ship on calm water appears as a very bright spot against a dark background, which is why it's useful for vessel detection.
- ==Sensitive to change==: Comparing two SAR passes of the same area ([[InSAR]]) lets you detect millimeter-scale ground deformation, which is used for earthquake/volcano monitoring and subsidence tracking.
- ==Detects structure, not color==: The signal encodes surface roughness, geometry, and dielectric properties (water content, metal).
	- ==Tradeoff:== SAR imagery looks nothing like a photograph; it takes practice to interpret, and artifacts like speckle noise require specific preprocessing before ML models can use it!

Common SAR satellites:
- [[Sentinel|Sentinel-1]] ([[European Space Agency|ESA]])
- [[RADARSAT]] (Canada)
- [[TerraSAR-X]] (Commercial, high-res)