---
aliases:
  - GPR
---
A near-surface method that uses radar pulses to image the subsurface, finding layer boundaries, objects, voids, changes in material.
- Platform types:
	- Ground-based: Drag or wheel antenna along surface for very high resolution, slow coverage
	- Vehicle-mounted: For road survey, utility mapping at driving speed
	- Airborne GPR: Used for measuring ice thickness in Antarctica (CreSIS, Operation IceBridge)
	- Drone-mounted: Emerging, small antennas, low altitude, for archaeological survey or landmine detection
	- Spaceborne: ==Not really possible==
		- The atmosphere absorbs the frequencies needed for useful ground penetration
		- MARSIS and SHRAD on Mars orbiters is an exception, but these work because MArs has no ionosphere/water vapor interference.

Key physics:
- The EM wave velocity in a material depends on its [[Dielectric Permittivity]] (ε): 
```
  v = c / √ε
```
- Water has a very high permittivity (~80), dramatically slowing radar and causing strong reflections.
	- This is why Wet soil = strong signal, shallow penetration
	- Dry sand = weak signal, deep penetration
	- Water table: Very bright reflector

GPR data format is the ==radargram==, a 2D cross-section image where:
- X-axis: Distance along the survey line
- Y: axis: Two-way travel time (proxy for depth)
- Brightness: Reflection strength

Same fundamental tradeoff as [[Synthetic Aperture Radar|SAR]] bands (e.g. [[Synthetic Aperture Radar|X-Band]], [[Synthetic Aperture Radar|P-Band]]), frequency governs penetration versus resolution:

![[Pasted image 20260419195526.png]]
Low frequency means deep but blurry, while high frequency means shallow but detailed.






