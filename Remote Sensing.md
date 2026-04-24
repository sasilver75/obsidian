---
aliases:
  - Earth Observation
  - EO
---
Note: This also serves as a dumping bucket for video notes, etc.

The acquisition of information about an object or phenomenon without making physical contact with the object, in contrast to "in situ" or on-site observation.
- Generally refers to the ==use of satellite or airborne sensor technologies to detect and classify objects on earth==.
	- In 2008, more than 150 Earth observation satellites were in orbit, rand by 2021, that total had grown to over 950. By 2023, there were 1,167.

Fundamental Principle of EO: All objects above absolute zero emit electromagnetic radiation.

It has military, intelligence, commercial, economic, planning, and humanitarian applications, among others.


> Comparison with [[Remote Sensing]]:
- EO: The gathering of information about the Earth
- RS: Gathering information about making contact
- So they are not the same! EO can or can not use RS; I typically talk about the intersection of the two, here.

![[Pasted image 20260421212101.png]]

Can be split into:
- **==**Active== remote sensing: When a ==signal is emitted by a sensor== on satellite/aircraft, and the reflection is detected by the sensor.
	- Emits energy to scan objects and areas whereupon a sensor then detects and measures the radiation that is reflected or backscattered from the target.
	- [[RADAR]] and [[Light Detection and Ranging|LiDAR]] are examples active remote sensing where the time delay between emission and return is measured, establishing the location/speed/direction of an object.
	- There are six main types of active instruments in [[National Aeronautics and Space Administration|NASA]]'s earth science data:
		1. Laser Altimeter: Uses LiDAR to measure the height of the platform (spacecraft, aircraft) above the surface, as a means of determining the topography of the underlying surface.
		2. [[Light Detection and Ranging|LiDAR]]: Uses a laser radar to transmit a light pulse and a receiver with sensitive detectors to measure the backscattered or reflected light. Distance to object is determined by recording time ∆.
		3. [[RADAR]]: Emits microwave radiation in a series of pulses from an antenna; some of it is reflected back towards the instrument. This backscattered microwave radiation is detected, measured, and timed. Distance to target is determined by recording time ∆. A two dimensional image of the surface can be produced as the instrument passes by. [[Synthetic Aperture Radar]] (SAR) is a radar technique.
		4. Ranging Instruments: Devices that measure the distance between the instrument and a target object. Radars and altimeters work by determining the time a transmitted pulse (microwaves or light) takes to reflect from a target and return to the instrument.
		5. Scatterometer: A high-frequency microwave radar designed specifically to measure backscattered radiation.
		6. Sounder: An instrument that measures the vertical distribution of precipitation and other atmospheric characteristics like temperature, humidity, and cloud composition.
- ==**Passive**== remote sensing: When the ==reflection of sunlight is detected== by the sensor.
	- Gathers radiation emitted or reflected by the object or surrounding areas; reflected sunlight is the most common source of radiation.
	- Includes film photography, infrared, radiometers.
	- The primary source of energy observed by such instruments is the Sun 🌞
		- ==The amount of the Sun's energy that is reflected depends on the roughness of the surface and its [[Albedo]]==, which is how well a surface reflects light instead of absorbing it.
	- Most passive instruments operate in the visible, infrared, thermal infrared, and microwave portions of the [[Electromagnetic Spectrum]], and most cannot penetrate dense cloud cover.
	- There are several categories of passive instruments in NASA's repertoire:
		1. Accelerometer: Measures acceleration; there are two types:
			1. Measures translational accelerations (changes in linear motions in one or more dimensions)
			2. Measures angular accelerations (changes in rotation rate per unit time)
		2. Hyperspectral Radiometer: Advanced multispectral instrument that detects hundreds of very narrow spectral bands throughout the visible, near-infrared, and mid-infrared portions of the EM. Facilitates fine discrimination between targets based on target *spectral response* in each of the narrow bands.
		3. Imaging Radiometer: Has a scanning capability to provide a two-dimensional array of pixels from which an image may be produced. 
		4. Sounder: Measures vertical distributions of atmospheric parameters such as temperature, pressure, and composition from multispectral information.
		5. Spectrometer: Detects, measures, and analyzes the spectral content of incident electromagnetic radiation. Conventionally uses gratings or prisms to disperse the radiation for spectral discrimination.
		6. Spectroradiometer: Measures the intensity of radiation in multiple wavelength bands. Many times, the bands are of high-spectral resolution, designed for remote sensing specific geophysical parameters.


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
	- The ==Orbital period== is how long one orbit takes.


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

![[Pasted image 20260422121153.png]]
- Raw is still useful; you want imagery as soon as possible for disaster relief/aid, so that you can help people impacted by natural disaster, war, etc.
	- Hasn't been aligned somewhere on the earth. You may know the satellite telemetry (position, angle, approximate), so you can roughly locate the image on earth, within a kilometer or so.
- Typically, unless you're doing immediate response applications, you wait for a L1 or L2 product
	- [[Top-of-Atmosphere]]: Typically an image that's been geometrically corrected and projected onto a [[Digital Elevation Model|DEM]], precisely geolocated. Has a little "tinge" to it, compared to the [[Surface Reflectance]] product.
	- [[Surface Reflectance]] has a bunch of preprocessing effects to do atmospheric correction (removing thin clouds, etc.). Useful for surface applications, but if you're studying the atmosphere itself, you'd typically rather do top-of-atmosphere.
- Level 3 products are so-called derived products, classified images created from satellite imagery..





# Common Remote Sensing Tasks
- [[Change Detection]]
	- ==Detecting *what changed*== between two or more observations of the same area.
	- The simplest form is *band differencing*, subtracting image A from B, with large values indicating change.  More sophisticated approaches use spectral indices (e.g. [[Normalized Difference Vegetation Index|NDVI]]), machine learning classifiers trained on change/no-change examples, or coherence analysis in SAR.
	- Hard parts: Normalizing for atmospheric or illumination differences between the dates, distinguishing real change from sensor noises, and handling seasonal variation (a forest looks different in summer vs winter without having "changed").
	- Applications: Deforestation Monitoring, Urban Expansion, Disaster Damage Assessment, Crop Monitoring.
- [[Land Cover]] Classification
	- ==Assigning every pixel to a class== (forest, water, urban, cropland, bare soil, etc.)
	- Approaches range from a simple (thresholding NDVI) to traditional ML (random forest on spectral + texture features), to deep learning (U-Net and variants for semantic segmentation).
	- The distinction between *land cover* (what's there) and *land use* (what it's being used for) matters; a parking lot and airport are both impervious surfaces but have different land uses.
- [[Object Detection]] and [[Semantic Segmentation]]
	- ==Detecting and delineating specific objects== (buildings, roads, vehicles, ships, solar panels, swimming pools). Usually deep learning now, typically trained on high-resolution commercial imagery. 
	- Building footprint extraction is the canonical task, while vehicle detection from high-resolution imagery is a defense/intelligence application. Ship detection from SAR is used for maritime monitoring.
	- The challenge is that objects of interest are tiny relative to scene size, training data is expensive to label, and performance varies across geographic regions and image conditions.
- [[Spectral Unmixing]]
	- Most pixels in moderate-resolution imagery (eg 30m [[Landsat]]) contain multiple materials. A pixel might be 60% forest, 30% soil, 10% shadow.
	- Spectral unmixing decomposes pixels into fractional abundance of pure spectral signatures called ***endmembers***.
	- ==Rather than classifying each pixel as one class, you get a continuous mixture.==
	- Useful for vegetation fraction mapping, mineral mapping, etc. Requires knowing or estimating the endmember spectra, which is often the hardest part.
- [[Time-Series Analysis]]
	- ==Analyzing *how* pixel values change over time==, rather than looking at single scenes.
	- Useful in Phenology (tracking vegetation growth cycles; when are plants planted/harvested?), trend detection (is the forest getting greener?), anomaly detection (drought stress, fire risk).
	- The challenge is that available time series are gappy (clouds, revisit time) and noisy (atmospheric variability), requiring robust statistical methods.
- [[Digital Elevation Model]] generation
	- ==Deriving terrain elevation from imagery==.
	- Comes in two flavors:
		- [[Digital Surface Model]]: Includes buildings/vegetation
		- [[Digital Terrain Model]]: : Bare earth only; requires filtering out above-ground objects
	- Methods include:
		- [[Photogrammetry|Stereophotogrammetry]] (using two optical images from different angles, matching features, and triangulating elevation from parallax)
		- [[InSAR]] (interfering two SAR acquisitions to get elevation; [[Shuttle Radar Topography Mission|SRTM]] was created this way)
		- [[Light Detection and Ranging|LiDAR]] (using active ranging; the most accurate, expensive, typically airborne; point clouds are processed into DEMs)
		- Structure from Motion (SfM): Similar to photogrammetry, but uses *many* overlapping images; common with drone imagery.
- Vegetation Analysis
	- Beyond simple NDVI, common tasks include:
		- Biomass estimation, important for carbon accounting
		- Species classification (typically tree species)
		- Forest structure (canopy height, leaf area index, crown closure; often from LiDAR)
		- Crop type mapping (which crops are planted in which fields?)
		- Stress detection (detecting drought stress, pest damage, disease before it's visible to the naked eye, using red-edge and SWIR bands)
- Water Analysis
	- Water body mapping (NDWI threshold is simple and works, Flood mapping sometimes uses SAR, since water appears dark)
	- Water quality: Turbidity, chlorophyll concentration, algal bloom detection
	- Bathymetry: Estimating shallow water depth from optical imagery using water's spectral absorption properties
	- Coastal changes: Shoreline migration, erosion monitoring
- Urban Analysis
	- Building footprint etraction
	- Urban heat island mapping
	- Impervious surface mapping (important for stormwater modeling)
	- 3D city models (combining building footprints with height data from stereo or LiDAR)
	- Informal settlement mapping (detecting slums, informal housing from texture)
- Atmospheric and Climate Applications
	- Aerosol and optical depth retrieval (measuring atmospheric particulates)
	- Cloud properties (cloud top height, optical thickness ice vs water phase)
	- Sear surface temperature
	- Fire detection and burned area maping
	- Snow cover and ice extent

==Each remote sensing product above *NEEDS VALIDATION==*
- Typically you collect reference data (visit fields, get high-resolution imagery, use existing maps) at  sampled location, and compare to the classified/derived product.
	- Compute confusion matrices, report overall accuracy, per-class precision/recall, and kappa coefficient.
- The sampling design matters enormously: Whether random sampling, stratified sampling, and how you handle class imbalance all affect whether your accuracy estimate is meaningful. 
	- This is often underemphasized but a remote sensing product *without rigorous accuracy assessment is scientifically incomplete!*


# Essential Variables (EVs) (critical for observing given facets)
- EVs are variables that are known to be critical for observing and monitoring a given facet of the Earth system.
- Climate
	- The Global Climate Observing System (GCOS) has identified as set of 54 atmosphere, land, and ocean variables as Essential Climate Variables (ECVs). Has atmosphere ECVs, land ECVs, and ocean ECVs.
![[Pasted image 20260416134228.png]]
- Ocean
	- According to UNESCO's Global Ocean Observing System (GOOS), critical ocean processes... include distribution and transport of heat, salt, and other water properties, exchanges of heat, momentum, freshwater, and gasses... and has other information on Essential Ocean Variables (EOVs).
![[Pasted image 20260416134358.png]]
- Biodiversity
	- Current species suggest that Earth is home to at least 2 million to 6 billion species; Essential Biodiversity Variables (EBVs) capture the essential dimensions of biodiversity so that it can be monitored.
	- The Group on Earth Observations Biodiversity Observation Network (GEO BON) has identified six *classes* of EBVs:
		- One for genetics
		- Two for species
		- Three for communities and ecosystems
![[Pasted image 20260416134548.png]]
- Geodiversity
	- The previous categories don't take into account consideration of the *abiotic surface and subsurface geology, mgeomorphology, and pedology (soil science) of an area.*
	- Essential Geodiversity Variables (EGVs) play a significant role in the availability of natural resources and often underpin the other essential variable frameworks.
		- Geodiversity has significant effects on biodiversity. Geodiversity directly influences the resources available for species, including energy, nutrients, and water. Geodiversity also plays a major role in the carbon, nitrogen, phosphorus, and sulfur cycles and their interactions with other Earth systems
	- For example, mapping the distribution of a species if difficult, as it's impossible to use remote sensing for direct detection of many taxa, so scientists often rely on geodiversity variables that are known to constrain where a species is most likely to survive and thrive as a proxy variable to estimate likely distribution.
- Agriculture
	- GEO's Global Agriculture Monitoring (GEOGLAM) is developing an Essential Agricultural Variables (EAV) to define the minimum set of variables in order to reinforce the international community's capacity to produce and distribute relevant, timely, and accurate information on agricultural land use and production at national, regional, and global scales.




# Notable Commercial Earth Observation Companies
- Optical Imaging
	- [[Planet Labs|Planet]]: Largest constellation, daily global coverage, good for change detection over time
	- [[Maxar]]: Highest commercial resolution, dominant in US defense/intelligence
	- [[Satellogic]]: Low-cost high-res and [[Hyperspectral]], disrupting on price
	- [[BlackSky]]: High-revisit tasking of specific sites with integrated real-time analytics
	- [[Airbus Pleiades Neo]]: European 30cm competitor to Maxar, strong commercial/gov 
	- ([[Planet Labs|SkySat]]): Planet's 50cm tasking capability, acq. in 2017 by Planet
- [[Synthetic Aperture Radar]] (SAR)
	- [[ICEYE]]: Finnish SAR leader, cloud-penetrating, strong in maritime and disaster response
	- [[Capella Space]]: US SAR, defense-focused, sub-50cm resolution
	- [[Umbra]]: Pushing SAR resolution limits (~16cm) with growing defense contracts
	- [[Synspective]]: Japanese SAR operator with expanding constellation
- Signals Intelligence
	- [[HawkEye 360]]: Detects and geolocates radio frequency emissions, ship tracking, [[Automatic Identification System|AIS]] spoofing detection
	- [[Spire Global]]: GPS radio occultation for weather + AIS ship tracking + RF monitoring
- Hyperspectral
	- [[Pixxel]]: Indian startup, hyperspectral constellation for agriculture, environmental monitoring.
	- [[Planet Labs|Tanager]] (Planet): Methane/Co2 detection from space, climate-focused
- Platforms/Marketplaces
	- [[UP42]]: Data marketplace aggregating imagery from multiple providers
	- [[Maxar SecureWatch]]: Subscription imagery platform widely used in enterprise/government
		- Replaced by [[Maxar Geospatial Platform]]



![[Pasted image 20260418155521.png]]

____________-

![[Pasted image 20260422195903.png]]
(These numbers are in the [[Google Earth Engine|GEE]] data catalog, counts of images from different satellites; graph shows continue to see more sensors/data being added over time.)
- We have about 100k new [[Sentinel|Sentinel-2]] images every week!

Challenge is now not the access, but the translation of these images into actionable insights!



_________________
[GeoAwesome: GeoAI Hype Cycle 2026 Panel](https://youtu.be/vyWcwFlcO3c)

Q: What is overhyped? What is underhyped?
A, George Lawrence @ [[UP42]]: Overhyped are smart satellites planning and acting on their own in space, having AI on satellites to task themselves, etc. Getting satellites off the ground is expensive; you want light satellites in the air, not heavy ones, and it's difficult to maintain satellites in space as well. But what I think is underrated... there's so much more that you can do with EO and GeoAI than defense, and in my work I see so many examples of this. Assessing a carbon above/below ground, monitoring environment, etc... So many GeoAI applications that the average person doesn't even appreciate... as people become more aware of what this technology can do, there will be challenges around confidentiality.
A, Bill Greer, Cofounder [[Common Space]]:  I think AI is good for somethings... but I'll leave that to the manufacturing engineer folks... my hype cycle thing.. is that I'm unconvinced by most of the AI stuff until it can prove that it's better than statistical methods or as good... there's huge potential in lowering latency and cost and increasing access. Selective compressive of imagery before downlinking is fine.... for capacity building, that's where I hope it comes in the most, getting more people using the data in statistically accurate and good ways, but opening it up to users... is something that I'm excite about.
A, Madeline Lisaius: I think satellite embeddings can offer value in many of these things... The whole premise of my thesis was how to make satellite data more accessible to people using embeddings. I see in many cases embeddings being a framework to do so, and I also think in many cases they're still overhyped. How can we help people who would benefit from this data most?

Q: What are some of those... use cases that are really moving towards genuine productivity and maturity that we'd like to see.
A: Madeline Liasus: My expertise is in the foundation model area, that's where the edge of my knowledge is... when it comes to foundation models, what really excites me... is  that with the right foundation model and customization, we're able to match and beat the results that were made by custom feature engineering, and to do so with a reduction in cost and compute by 10-20x. So taking away the knowledge requirements of being able to generate custom features... and also comptue rquirements. But results are not 10-20x better in accuracy. 

Q: Requirements to train these models that you're seeing at UP42?
A, Gordon Lawrence: Our mission is to deliver data from different providers in a consistent format... there are three things to mention:
- Consistency: Lots of providers out there. But it's important to deliver data with consistent metadata. WE use [[SpatioTemporal Asset Catalog|STAC]] to do that at UP42, but it needs to go beyond metadata; there needs to be consisstencyi n the [[GeoTIFF]]s we delivery (consistent multispectral band ordering, for instance).
- [[Analysis Ready Data]]: We work with partners like Airbus, Nanospace, etc to release processing capabilities.... like [[Pansharpening]], Photoregistration ([[Georeference]]?), upsampling... We've been consisting 16-bit analytical images into 8-bit display-ready images...
	- At a danger of sounding like a demo, I was working with some super pixelated, very fuzzy data and we put it through some of these capabilities, and wended up with super sharp data, where you could recognize trees, railroad tracks, etc... and it was sharp and ready for analysis.
		- (He seems ill informed on this, IMO. superresolution is by definition confabulation)
- Accessibility: We're delivering [[Cloud-Optimized GeoTIFF|COG]]s, as well as streaming GeoTIFFs, almost Google Maps-like technology, so people can stream geospatial images for whatever application they've got
	- We make sure we're documented by console, API, and SDK... next might be an [[Model Context Protocol]] (MCP) server.
	- As well as access control things...


________________

[The Future of Geospatial AI (from Mmichaela Nadine Pacis, Marketing Manager @ Kili Technology, a data labeling tool company)](https://youtu.be/0sOs24APz8c)


....


_______









