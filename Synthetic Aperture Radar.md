---
aliases:
  - SAR
  - L-Band
  - S-Band
  - X-Band
  - C-Band
  - P-Band
  - HH
  - HV
  - VH
  - VV
  - Co-Pol
  - Cross-Pol
  - Stripmap
  - ScanSAR
  - Spotlight
  - TOPSAR
---
Resources:
- [NASA EarthData: Synthetic Aperture Radar](https://www.earthdata.nasa.gov/learn/earth-observation-data-basics/sar) (Unread, looks great)
- [Video: ICEEYE: The Value of SAR](https://youtu.be/GwyldSNlefk)

A [[Remote Sensing]] technique that uses radar signals instead of visible light to image the Earth's surface.
- The heritage of SAR is that it was usually used for intelligence, or very large civil programs (e.g. NASA, CSA, ESA).
- Just over a decade ago, a few very capable SAR satellites [[TerraSAR-X]], [[RADARSAT|RADARSAT-2]], [[COSMO-SkyMed]]. These are public/private partnerships and massive satellites, over 1,000kg. 
	- [[ICEYE]] revolution is that they have a small SAR satellite constellation funded primarily by VC, with mass of about 100kg and much cheaper, so there can be more.


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



# SAR Bands
- In optical, "bands" mean wavelength channels (red, [[Near Infrared|NIR]], [[Short-Wave Infrared|SWIR]]). In SAR, "bands" refers to the microwave frequency that the radar transmits, and the data you get back is about how the surface scatters that signal, not reflected sunlight.
- ==Key Intuition:== ==Wavelength ~= penetration depth==
	- Shorter wavelength: bounces off surfaces, good for texture, structure, surface motion
	- Longer wavelength: Penetrates through, good for what's underneath (canopy, soil, ice)
- ==X-Band== Radar: ~3cm wavelength, 8-12GHz, best for fine detail (urban, sea ice surface)
	- [[TerraSAR-X]], Cosmo-SkyMed, [[Sentinel|Sentinel-1]] (partly)
- ==C-Band== Radar: ~5.6cm wavelength, 4-8GHz, best for surface change, floods, agriculture
	- [[Sentinel|Sentinel-1]], [[RADARSAT|RADARSAT-2]]
- ==S-Band== Radar: ~10cm wavelength, 2-4GHz, best for (transitional), soil moisture
	- [[NISAR]]
- ==L-Band== Radar: ~24cm wavelength, 1-2GHz, best for forest biomass, soil, subsurface
	- [[PALSAR|PALSAR-2]], [[NISAR]]
- ==P-Band== Radar: ~70cm wavelength, 0.3-1GHz, best for deep forest biomass, root zone
	- Key sensors: BIOMASS (ESA, 2024)

# SAR Polarization
- SAR has various polarization channels:
	- ==HH==: Transmit horizontal, receive horizontal
	- ==VV==: Transmit vertical, receive vertical
	- ==HV==: Transmit horizontal, receive vertical (cross-pol)
	- ==VH==: Transmit vertical, receive horizontal (cross-pol)
- ==Co-pol== (HH, VV): Surface and double-bounce scattering (ground, buildings, water)
- ==Cross-pol== (HV, VH): Volume Scattering (vegetation canopy, rough surfaces)

# Scattering Mechanisms
Decomposing a quad-pol image into these three mechanisms (Pauli, Freeman-Durden decomposition) is a core SAR analysis technique
- ==Surface scattering==: Flat/rough bare surfaces, strong in VV, bare soil, calm water
- ==Double-bounce==: Two perpendicular surfaces (wall, ground), HH, urban, flooded trees
- ==Volume scattering==: Many random scatters (leaves, branches), HV, dense vegetation

# [[Interferometry]] ([[InSAR]]): The Phase Dimension
- Beyond intensity, SAR records ==phase==, the position in the wave cycle when backscatter is returned.
- Two passes over the same area let you compute phase difference, which can help you deduce surface deformation (change) at millimeter-level precision.

Common SAR satellites:
- [[Sentinel|Sentinel-1]] ([[European Space Agency|ESA]])
- [[RADARSAT]] (Canada)
- [[TerraSAR-X]] (Commercial, high-res)
- [[PALSAR]] (Japanese, L-band)

![[Pasted image 20260419191046.png]]
Above:
- Four SAR imaging modes
	- ==Stripmap==: The default, simplest mode. Antenna points at a fixed angle off to the side, satellite flies forward, sensor illuminates a continuous strip. Like dragging a flashlight beam along the ground at a fixed width.
	- ==ScanSAR==: Antenna electronically sweeps across multiple adjacent strips (subswaths) during the pass, trading resolution for width. You cover more ground per pass, but each subswath gets illuminated for less time, so fewer radar pulses hit each point, resulting in coarser resolution. This is how [[Sentinel|Sentinel-1]]'s wide-area mode works.
	- ==Spotlight==: The antenna rotates to keep dwelling on one target as the satellite passes overhead. That one patch gets illuminated for much longer, many more pulses result in finer resolution, but you can only do this for one small area at a time, sacrificing swath entirely. Used when you need maximum detail on a specific point of interest.
	- ==TOPSAR:== A hybrid that fixes ScanSAR's main problem. Standard ScanSAR has *scalloping*, intensity variations between subswathes because edge areas get fewer pulses than center areas. TOPSAR sweeps the antenna in *bursts* with controlled timing, spreading illumination more evenly across each subswath. The result is a wide swath like ScanSAR but without the scalloping artifacts, and compatible with [[InSAR]], which standard ScanSAR isn't!
- Core tradeoff: ==The longer you illuminate a patch, the finer your resolution, but dwelling on one patch means you're not illuminating anything else!==


![[Pasted image 20260420213305.png]]
Above: Showing [[Synthetic Aperture Radar|L-Band]] vs [[Synthetic Aperture Radar|C-Band]] vs [[Synthetic Aperture Radar|X-Band]]


__________

[ICEYE: The Value of SAR](https://www.youtube.com/watch?v=GwyldSNlefk)

We're going to send discrete pulses of radar to the gorund:
- We measure the range to the ground and the strength of the echo
- When the radar pulse hits the ground, it reflects in many directions, which is the backscatter. We capture the backscatter, and the intensity of the received backscatter is the brightness of the pixel.
- The ==Pulse Repetition Frequency (PRF)== is the rate at which we send these pulses. For an aircraft it's many hundreds a second, for a satellite, it might be 2-3,000 pulses per second.
- For optical cameras, looking directly down at the ground gives you the best resolution. For SAR, that doesn't work. SAR measures *range*. If we illuminated the ground vertically, we could have two points with exactly the same range in different places. Those points for a given line would be placed at the same pixel location on the image, and it wouldn't work.
	- ![[Pasted image 20260420221421.png]]
	- So for SAR, we have to be side-looking. In this configuration, every point on the ground has a discrete range, and therefore all the pixels would be in different locations based on their range. 
	- ![[Pasted image 20260420221431.png]]


Most SARs illuminate *broadside,* meaning 90 degrees to the flight path. Some sensors can squint, meaning illuminate the ground forward or behind the broadside. This is called squinting, and the angel is the ==squint angle==.
![[Pasted image 20260420221645.png]]

The grazing angle is the angle between the line of sight ray and the tangent to the ground.


The incidence angle is the angle between the line of sight ray and the local vertical.

Different people use these two measures: Some say incidence angle, some say grazing angle. So for the vague "look angle," you have to understand which they're talking about.

Notice that the two of them form a right angle, so a 30 degree grazing angle is equivalent ot a 60 degree incidence angle.



