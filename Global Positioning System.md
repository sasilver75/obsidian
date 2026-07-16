---
aliases:
  - GPS
---
ReferenceS:
- Article: [How the Heck Does GPS Work? (Interactive animations)](https://perthirtysix.com/how-the-heck-does-gps-work)


A [[Global Navigation Satellite System]] (GNSS).
The US-Operated Satellite [[Constellation]] that provides positioning, navigation, and timing (PNT) anywhere on Earth.
- Ground truth for [[Remote Sensing]]; GPS-tagged field observations validate satellite classifications. Used for [[Georeference|Georeferencing]] of imagery/point clouds.

# Accuracy
- Consumer GPS is accurate to ~3-5m
- WAAS/SBAS (involves differential correction via ground stations): ~1m
- [[Real-Time Kinematic Positioning]]: centimeter level, used in surveying, precision ag, AVs
- [[Precise Point Positioning]] (PPP): Also centimeter level, but slower to converge.

# How it works:
- 24+ satellites in [[Medium Earth Orbit]] (~22,200km), arranged so that 4+ are always visible from any point.
- Each satellite continuously broadcasts its position + precise atomic clock timestamp.
- Your receiver measures the *time of flight* of signals from multiple satellites
- With 3 satellites you can get your 2D position; a fourth satellite resolves altitude and corrects for receiver clock error.
	- This is called *trilateration* (not triangulation; you're measuring distances, not angles)
	- Time of flight from satellites -> distance spheres -> where they intersect is you



> A lot of people who don't use it... might be oblivious to things that happen in space (solar flares, coronal holes, sunspots, F10.7 cm Radio Emissions, Solar EUV irradiance, coronal mass ejections, solar wind, solar radiation storms, geomagnetic storms, ground induced currents, ionospheric scintillation, etc.) that degrade GPS... Satellites orbit the earth, and there are so many you uses for navigation... sometimes the triangulatio n gets too close and you have ==dilution of precision==. Note also that the different [[Global Navigation Satellite System|GNSS]] frequencies are pretty well-stacked over eachother, so if you're a good guy trying to jam [[Globalnaya Navigazionnaya Sputnikovaya Sistema|GLONASS]], you might also inherently jam your own stuff! Fratricide via jamming... and a lot of drone system are not designed to take off without a GPS lock. [[Visual Positioning System]]s can help you get around these.

![[Pasted image 20260715145839.png]]

![[Pasted image 20260715145745.png]]



![[Pasted image 20260715145649.png]]