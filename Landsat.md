---
aliases:
  - Landsat-1
  - Landsat-2
  - Landsat-3
  - Landsat-4
  - Landsat-5
  - Landsat-6
  - Landsat-7
  - Landsat-8
  - Landsat-9
  - Landsat Next
  - Worldwide Reference System
---


The longest running enterprise that has acquired satellite imagery of earth, first images released in 1972.

A joint [[National Aeronautics and Space Administration|NASA]]/[[United States Geological Survey]] program.
- The first was launched in 1972, the most recent (Landsat 9) launched in September 2021.

Images are used for agriculture, cartography, geology, forestry, regional planning, surveillance, and education.

Landsat 7 data has 8 spectral bands with 15-60m resolution, and 16 day temporal resolution.

![[Pasted image 20260415205517.png]]



_________________________ 

Adam Stewart AI4EO (2026)

Longest running EO program, from July 1972 until today.
- A range of resolutions; older ones are 60m/pix, newer are 30m/pix or so. But it's a long history, so you can study long-term changes on the Earth's surface, which is what we care about if we're doing 
- Notice that some of these satellites died early; Landsat 6 failed to reach orbit and crashed down to earth, whereas 4,5,7 had extremely long lifespans, longer than the creators thought the satellite would last 
![[Pasted image 20260422120553.png]]
- The cross-hatched stuff indicate partiail or complete sensor failure
	- In Landsat 7, a scalnline corrector (controls position of camera) broke, so you get weird images that have stripes.
	- But the stripes that have data are still useful!
![[Pasted image 20260422120745.png]]
- Different satellites carry one or more sensors on board.
	- Landsats 1-3 carried two sensors: The Return Bean Vidicon (kind of a regular RGB camera) and the Multispectral Scanner (more of a [[Multispectral]] imaging source). 
	- In 4-5 they ditched the RBV and replaced it with the TM.
	- In 6, they had an enhanced version, but it crashed.
	- In 7, they had enhanced plus, an even better one with higher res!
	- In 8-9, there's an optical and thermal sensor.
	- Landsat NEXT, the new one, will be launched in a few years, will have significantly more spectral bands to match [[Sentinel|Sentinel-2]], at much higher resolution to match Sentinel 2!

![[Pasted image 20260422120956.png]]
Above: Can see the spectral bands for Landsat Next; these are lined up with the atmospheric transmissivity of different wavelengths.



![[Pasted image 20260422122421.png]]
Above: [[Landsat|Worldwide Reference System]]
- See that for some locations, you can get coverage 4x during a 2 week period (corners), whereas other parts will get coverage 2x, whereas other locations will only get 1x every two weeks.
- How far you are from the equator governs how much overlap there will be for different images.
- When you download images, you can say: "I'm looking at path 30, row 38"


