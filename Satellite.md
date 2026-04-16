

# Terms of Art:
- "==Aquisition==": The act of a satellite capturing imagery over a specific location
- "==Tasking==": Instructing a satellite to acquire imagery of a specific location at a specific time.
	- Commercial satellites like [[Maxar]]'s are ==taskable==; you pay to point them where you want. Contrast this with ==survey mode== satellites like [[Planet Labs|Planet]]'s Doves, which just image everything below them continuously without being directed at specific targets.
- "==Revisit Time / Revisit Rate==": How frequently a satellite or constellation passes over the same ground location. A single [[Sentinel|Sentinel-2]] satellite has a 10-day revisit, but with both satellites (2A + 2B) combined, it's ~5 days. High revisit is critical for time-series and change detection work.
- "==Swath Width==": The width of the strip of ground a satellite images on each pass. A wide swath means more area is covered per pass, but often at lower resolution.
- "==Ground Sample Distance==" ([[Ground Sample Distance|GSD]]): The most precise term for what people loosely call "resolution"; The real-world size represented by one pixel. WorldView-3's 0.31m GSD means each pixel covers a 31cm by 31cm square on the ground.
	- GSD varies slightly with off-nadir angles; the further the satellite tilts from straight down, the larger the effective GSD.
- "==Nadir==" vs "==Off-Nadir==":
	- Nadir is the the point directly below the satellite; a straight-down look angle.
	- Off-nadir is when the satellite tilts to image a target to the "side"; taskable satellites can tilt significantly (WorldView-3 up to ~45 degrees) to increase revisit flexibility or collect stereo pairs.
		- Results in larger effective GSD, more geometric distortion, and more visible building/terrain lean.
- "==Footprint=="
	- The geographic area (polygon) covered by a single scene or image acquisition. Also used for a satellite's instantaneous field of view projected onto the ground.
- "==Scene== / ==Strip== / [[Tile]]"
	- Strip: The continuous ribbon of imagery collected during one pass, before being cut into scenes.
	- Scene: A discrete image product, usually with a fixed-size geoegraphic chunk (e.g. 100km x 100km tiles)
	- Tile: Often used interchangably with scene, or specifically for pre-defined grid cells (e.g. XYZ web tiles, MGRS tiles, H3 cells)
- ==[[Orthoimage|Orthorectification]]==
	- Correcting for terrain distortion and sensor geometry so the image is planimetrically accurate.
- [[Analysis Ready Data]] (ARD)
	- Imagery preprocessed to a standard that you can use it directly without additional radiometric/geometric correction. 
	- Typically means:
		- Orthorectified
		- Atmospherically corrected to surface reflectance
		- Cloud-masked
		- In a consistent coordinate system
- [[Top-of-Atmosphere]] (TOA) (c.f. Surface Reflectance)
	- Raw radiance with minimal correction - includes atmospheric scattering and absorption effects.
- [[Surface Reflectance]] (SR) (c.f. Top of Atmosphere)
	- Atmospherically corrected to represent actual ground surface reflectance.
- [[Panchromatic]] (Pan): Single broad band covering visible light.
- [[Multispectral]]: A handful of discrete bands (typically 4-12), e.g. RGB + [[Near Infrared]] (NIR) + [[Short-Wave Infrared]] (SWIR).
- [[Hyperspectral]]: Hundreds of narrow contiguous bands capturing a full spectral signature; used for mineral mapping, precision agriculture, water quality.
- [[Pansharpening]]
	- Fusing a high-resolution panchromatic band with lower-resolution multispectral bands to produce a high-res color image.
- [[Cloud-Optimized GeoTIFF]] (COG)
	- A GeoTIFF structured so that HTTP range requests can fetch only the spatial subset and resolution level you need, without downloading the whole file.
	- The foundation of modern cloud geospatial workflows.
- [[STAC|SpatioTemporal Asset Catalog]] (STAC)
	- A standard JSON spec for cataloging geospatial assets with spatial extent, time range, and links to actual files. 
	- Lets you query: "Give me all [[Sentinel|Sentinel-2]] scenes over this bbox with <20% cloud cover between these dates" via a standard API.
	- [[Microsoft Planetary Computer]] and [[Element84 Earth Search]] are the major public STAC holds.
- Area of Interest (AOI)
	- The geographic region you're working on. Universally used abbreviation in geospatial work.
- [[Digital Elevation Model]] (DEM)
	- Generic term for *any* elevation raster
- [[Digital Surface Model]] (DSM)
	- Elevation of the top surface - includes buildings, trees, everything above bare earth.
	- ==A DSM is a DEM of a surface== (including buildings, trees, powerlines)
- [[Digital Terrain Model]] (DTM)
	- Bare-earth elevation with buildings/vegetation removed.
	- ==A DTM is a DEM of the bare earth== (buildings/vegetation are filtered out)
- Radiometric Resolution
	- How many distinct intensity levels the sensor reads, in bit depth:
		- 8-bit = 256 levels
		- 12-bit = 4,095 levels
		- 16-bit = 65,536 levels
	- Higher radiometric resolution lets you distinguish subtle reflectance differences (important for vegetation indices, water quality)






![[Pasted image 20260415115045.png]]
Above: [Source](https://nanoavionics.com/blog/how-many-satellites-are-in-space/)

![[Pasted image 20260415115109.png]]
Above: [Source](https://nanoavionics.com/blog/how-many-satellites-are-in-space/)
- By the 2020s, small satellites comprised 94% of all satellite launches.