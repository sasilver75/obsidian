The acquisition of information about an object or phenomenon without making physical contact with the object, in contrast to "in situ" or on-site observation.
- Generally refers to the ==use of satellite or airborne sensor technologies to detect and classify objects on earth==.
	- In 2008, more than 150 Earth observation satellites were in orbit, rand by 2021, that total had grown to over 950. By 2023, there were 1,167.

It has military, intelligence, commercial, economic, planning, and humanitarian applications, among others.

Can be split into:
- **==**Active== remote sensing: When a ==signal is emitted by a sensor== on satellite/aircraft, and the reflection is detected by the sensor.
	- Emits energy to scan objects and areas whereupon a sensor then detects and measures the radiation that is reflected or backscattered from the target.
	- [[RADAR]] and [[Light Detection and Ranging|LiDAR]] are examples active remote sensing where the time delay between emission and return is measured, establishing the location/speed/direction of an object.
- ==**Passive**== remote sensing: When the ==reflection of sunlight is detected== by the sensor.
	- Gathers radiation emitted or reflected by the object or surrounding areas; reflected sunlight is the most common source of radiation.
	- Includes film photography, infrared, radiometers.


Data Characteristics:
- ==Spatial resolution==
	- The size of a pixel recorded in a raster image; typically pixels correspond to square areas ranging in side length from 1 to 1000 meters (3.3 to 3280.8 ft).
- ==Spectral resolution==
	- The bandwidth of the different frequency bands recorded; usually, this is related to the number of frequency bands recorded by the platform. Current [[Landsat]] collection is that of seven bands, including several in the infrared spectrum.
- ==Radiometric resolution==
	- The number of different intensities of radiation the sensor is able to distinguish. Typically ranges from 8 to 14 bits, corresponding to 256 levels of the gray scale and up to 16,384 intensities or "shades" of color, in each band.
- ==Temporal resolution==
	- The frequency of flyovers by the satellite or plane, which is mostly relevant in time-series studies or those requiring an averaged or *mosaic* image, as in deforesting monitoring.
		- Cloud cover over a given area or object makes it necessary to repeat the collection of said location.


A variety of corrections may need to be applied to images:
- ==Radiometric correction==: The illumination of objects on the Earth's surface is uneven because of different properties of the relief; this distortion can be corrected for.
- ==Topographic correction==: In rugged mountains, as a result of terrain, the effective illumination of pixels varies considerably. In remote sensing images, the pixel on the shady slope receives weak illumination and has a low radiance value, while a pixel on the sunny slope receives strong illumination and has high radiance values. This can seriously effect image information extraction accuracy in mountainous areas; topographic correction aims to eliminate this effect, recovering the true reflectivity or radiance of objects in horizontal conditions.
- ==Atmospheric correction==: Elimination of atmospheric haze by rescaling each frequency band so that its minimum value (usually realized in water bodies) corresponds to a pixel value of 0.


Generally speaking, in inverse sensing, the object or phenomenon of interest (the **state**) may not be directly measured, but there exists some other variable that can be detected and measured (the **observation**), which may be related to the object interest through a calculation.
- Like trying to determine the type of animal from its footprints
	- e.x.: While it's impossible to directly measure temperatures in the upper atmosphere, it is possible to measure the spectral emissions from a known chemical species (like carbon dioxide) in that region. The frequency of emissions can be related via thermodynamics to the temperature in that region.


### Data Processing Levels
A taxonomy of processing "levels" defined in 1986 by [[National Aeronautics and Space Administration|NASA]] as part of its Earth Observing System, and steadily adopted since then, both at NASA and elsewhere:
- 0:
	- Reconstructed, unprocessed instrument and payload data at full resolution, with any and all communication artifacts removed.
- 1a:
	- Reconstructed, unprocessed instrument data at full resolution, time-referenced and annotated with ancillary information (e.g. radiometric and geometric calibration coefficients and georeferencing parameters) computed and appended (but not applied) to the Level 0 data.
	- ==The most fundamental data record upon which all subsequent data sets are produced.==
- 1b:
	- Level 1a data that's been processed to sensor units (e.g. radar backscatter cross section, brightness temperature, etc.); not all instruments have Level 1b data.
- 2:
	- Derived geophysical variables (e.g. ocean wave height, soil moisture, ice concentration) at the same resolution as L1 source data.
	- ==The first level that is directly usable for most scientific applications; these datasets tend to be less voluminous than L1 data because they have been reduced temporally/spatially/spectrally.==
- 3:
	- Variables mapped on uniform spacetime grid scales, usually with some completeness and consistency (e.g. missing points interpolated, complete region mosaicked together from multiple orbits, etc.)
	- ==Generally smaller than lower level data sets, and thus can be dealt with without incurring a great deal of data handling overhead. Useful for many applications.==
- 4:
	- Model output or results from analyses of lower level data (i.e. variables that were not measured by the instruments, but instead derived from these measurements).







