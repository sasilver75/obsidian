

# Terms of Art:
- "==Aquisition==": The act of a satellite capturing imagery over a specific location
- "==Tasking==": Instructing a satellite to acquire imagery of a specific location at a specific time.
	- Commercial satellites like [[Maxar]]'s are ==taskable==; you pay to point them where you want. Contrast this with ==survey mode== satellites like [[Planet Labs|Planet]]'s Doves, which just image everything below them continuously without being directed at specific targets.
- "==Revisit Time / Revisit Rate==": How frequently a satellite or constellation passes over the same ground location. A single [[Sentinel|Sentinel-2]] satellite has a 10-day revisit, but with both satellites (2A + 2B) combined, it's ~5 days. High revisit is critical for time-series and change detection work.
	- Compare with "Orbital period"
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
- [[SpatioTemporal Asset Catalog|SpatioTemporal Asset Catalog]] (STAC)
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
- ==Radiometric Resolution==
	- How many distinct intensity levels the sensor reads, in bit depth:
		- 8-bit = 256 levels
		- 12-bit = 4,095 levels
		- 16-bit = 65,536 levels
	- Higher radiometric resolution lets you distinguish subtle reflectance differences (important for vegetation indices, water quality)
- ==Orbital Period==: How long one orbit takes
- ==Inclination==: The angle between the orbital plane and the equatorial plane. 0 degrees is an equatorial orbit, and 90 degrees is a polar orbit.
- ==Retrograde Orbit==: Orbiting in the opposite direction to Earth's rotation (east to west). Most [[Sun-Synchronous Orbit|Sun-Synchronous]] orbits are technically slightly retrograde (>90 deg inclination), which is what causes the orbital plane to precess in sync with the Sun.
- ==Prograde Orbit==: Orbiting in the same direction as Earth's rotation (west to east). Most satellites are prograde.
- ==Precession==: The slow rotation of an orbital plane over time due to Earth's equatorial bulge (Earth isn't a perfect sphere). Normally an annoyance that causes orbital drift. [[Sun-Synchronous Orbit|Sun-Synchronous]] orbits deliberately exploit precession: The inclination is tuned so that the plane precesses at exactly 360 degrees a year, keeping the satellite's crossing time synchronized with the sun.
- ==Ground track==: The path that the satellite traces on the Earth's surface over time. For a [[Sun-Synchronous Orbit|Sun-Synchronous]] satellite, this is a series of parallel descending/ascending strips.
- ==Look Angle==: The angle between Nadir (the point directly below the satellite) and where the sensor is pointing. Small look angles give better geometry, while large look angles increase distortion and atmosphere path length.
- ==Atmospheric drag==: Even at LEO altitudes, there's still trace atmosphere. Drag slowly decays orbits, requires periodic reboosts using onboard thrusters to maintain altitude. The reason why LEO satellites have shorter lifespans.
- ==J2 perturbation==: The dominant gravitational perturbation from Earth's equatorial bulge, causing orbital plane precession (the same effect that [[Sun-Synchronous Orbit|Sun-Synchronous]] orbits exploit).
- ==Solar radiation pressure==: Photons from the sun exert a tiny but perceptible force on satellites that matters for orbit determination and altitude control.
- ==Station Keeping==: The ongoing process of firing thrusters to correct for orbital perturbations (atmospheric drag, J2 perturbation, solar radiation pressure) and maintain the desired orbit.
	- [[Geostationary Orbit|Geostationary]] satellites need North-South and East-West station keeping.
	- [[Low Earth Orbit|LEO]] satellites mainly need altitude maintenance.
- ==Attitude==: The orientation of the satellite in space (roll, pitch, yaw). Not position, attitude is which way it's pointing. This is measured and controlled by ==ADCS systems== (Attitude Determination and Control System), which uses star trackers, sun sensors, magnetometers to determine orientation, and reaction wheels and magnetic torquers to adjust it.
	- Reaction wheels are spinning wheels inside the satellite; changing their spin rate transfers angular momentum to the satellite body rotating it. No fuel needed, purely electric.
- ==Walker Constellation==: A specific orbital configuration of multiple satellites designed to maximize coverage. Used by [[Global Positioning System|GPS]], [[Iridium Satellite Constellation|Iridium]], and many imaging constellations.
- ==Duty Cycle==: The fraction of time the sensor is actually imaging vs in other modes (downlinking, maneuvering, in eclipse). Power and thermal constraints limit how continuously a satellite can image.
- ==Eclipse==: When the satellite passes into Earth's shadow. No solar power, running on batteries, sensor usually off. LEO satellites experience ~15 eclipses per day, while Sun-synchronous orbits can be designed to be nearly eclipse-free.
- SAR-specific concepts:
	- ==Synthetic Aperture==: SAR simulates a very large antenna by combining radar returns collected over a long flight path. A longer synthetic aperture = finer along-track solution, which is why SAR can achieve high resolution from a small physical antenna.
	- ==Coherence==: How similar the radar phase signal is between two SAR acquisitions. High coherence means the surface hasn't changed, and low coherence (decorrelation) means change; used in [[InSAR]] change detection.
	- ==Backscatter==: How much radar energy bounces back to the sensor. Rough surfaces scatter more, smooth surfaces can act like mirrors and bounce energy away, appearing dark.
	- ==Polarization==: Radar can transmit and receive in horizontal (H) or vertical (V) polarization; Different polarization combinations (HH, VV, HV, VH) reveal different surface properties.
	- ==Interferometric SAR== (InSAR): Comparing the phase of two SAR acquisitions to measure surface deformation at centimeter or millimeter scale. Used for earthquake displacement, volcano inflation, glacier movement, subsidence.
- Radiometry concepts:
	- Irradiance: The power of electromagnetic radiation *incident* on a surface per unit area, in (W/m^2). "What hits the surface"
	- Radiance: Power emitted or reflected from a surface per unit area pr unit solid angel (W/m^2/sr). What the sensor measures.
	- Albedo: Fraction of incoming *solar radiation* reflected by a surface across *all directions* and wavelengths. Fresh snow: ~0.9, Ocean: ~0.06. A bulk surface property, *not* a directional measurement.
	- BRDF (Bidirectional Reflectance Distribution Function): How a surface reflects light differently depending on the illumination angle and viewing angle. A forest reflects very differently depending on if you look at it from directly above vs at an oblique angle.
	- Emissivity: How efficiently a surface radiates thermal energy compared to a perfect blackbody. Water is ~0.99, dry sand is ~0.84.




![[Pasted image 20260417124944.png]]





![[Pasted image 20260415115045.png]]
Above: [Source](https://nanoavionics.com/blog/how-many-satellites-are-in-space/)

![[Pasted image 20260415115109.png]]
Above: [Source](https://nanoavionics.com/blog/how-many-satellites-are-in-space/)
- By the 2020s, small satellites comprised 94% of all satellite launches.