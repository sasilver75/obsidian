---
aliases:
  - SRTM
---
An 11-day [[National Aeronautics and Space Administration|NASA]] space shuttle mission in Feburary 2000 that ==produced the first near-global, high-resolution [[Digital Elevation Model]]== of the Earth's land surface.
- One of the most impactful and widely-used datasets in all of geospatial science!
- Before SRTM, consistent global elevation just didn't exist! Despite being 25 years old and superseded by better prdouces, SRTM data is still everywhere.
- ==Successors and Alternatives==
	- ASTER GDEM: Derived from steo-optical imagery; Global, 30m, free, generally considered noisier than SRTM but covers polar regions that SRTM doesn't.
	- ALOS World 3D (AW3D30): 30m global, considered somwhat better than SRTM in many areas, free for research use.
	- [[Copernicus DEM]]: Derived from TanDEM-X, 30m global, released freely in 2021. Generally considered the *==best freely available global DEM now==*, with better accuracy than SRTM, more recent, better void filling. Increasingly the default choice for new work.
	- [[WorldDEM|TanDEM-X]]/[[WorldDEM]]: The full-resolution commercial product (12m) from the same mission as above; Not free, sold by Airbus.
	- [[LiDAR]]: Where it exists, it's far superior to any satellite-derived DEM. 1m or better resolution, true bare-earth capability. Coverage is patchy, mostly in developed countries, coastal areas, and areas with specific research needs.

Space Shuttle Endeavor flew with a modified radar system that used ==[[InSAR|Single-Pass InSAR]]== *two radar antennas separated by a 60-meter boom*  extended from the shuttle cargo bay, with one transmitting radar pulses and both receiving the returns.
- The slight geometric difference for the two produced interferometric phase measurements which could be converted to elevation!

Single-pass InSAR is important: Because both antennas were on the same platform at the same moment, there was ==no temporal decorrelation (surface didn't change between acquisitions), giving a clean, coherent phase measurement across almost all terrain types.

The products:
- SRTM1: 1 arc-second resolution (~30m at equator), released 2014
- SRTM3: 3 arc-second resolution (~90m at equator), released from the beginning

Both are [[Digital Surface Model]]s, capturing the top of *whatever is on the surface* (whether buildings, trees, or bare earth). In forested areas, this means that SRTM reflects canopy height.

SRTM has data voids, areas where the radar couldn't get a good measurement:
- Very steep terrain where radar shadowing occurred
- Smooth surfaces (water, sand) with low backscatter
- Some areas near the coverage limits



