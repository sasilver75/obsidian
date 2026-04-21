References:
- Video: [The Efficient Engineer: How to Build a Satellite](https://youtu.be/5voQfQOTem8) (Incredible overview of Satellite design!)
- Video: [How to build a satellite with Stuart Eves](https://www.youtube.com/watch?v=oWidvY7JzFE)  (More technical/engineer-oriented than Efficient Engineer)


Space is a near-perfect vacuum, but contains lots of high-energy radiation, so we need to ensure that the satellite can operate in this harsh environment!

The Copernicus project, Sentinel-2A/B satellites, produces 9.54TB of data each day. This amount accelerates as other companies enter the space. 

# Terms of Art:
- "==Aquisition==": The act of a satellite capturing imagery over a specific location
- "==Tasking==": Instructing a satellite to acquire imagery of a specific location at a specific time.
	- Commercial satellites like [[Maxar]]'s are ==taskable==; you pay to point them where you want. Contrast this with ==survey mode== satellites like [[Planet Labs|Planet]]'s Doves, which just image everything below them continuously without being directed at specific targets.
		- Note: "Systematic" is the inverse of "Taskable", from an operating mode perspective. "Survey mode" refers to a specific imaging mode, where the sensor sweeps a continuous strip along the ground track at a consistent resolution and swath width; they often coincide (i.e. [[Sentinel|Sentinel-1]] operating systematically and in stripmap mode (as survey mode is called in [[Synthetic Aperture Radar|SAR]])), but they *can* come apart: A tasked satellite like [[Maxar|WorldView]] can still use a stripmap-style collection mode, it just do so on request.
- "==Revisit Time / Revisit Rate==": How frequently a satellite or constellation passes over the same ground location. A single [[Sentinel|Sentinel-2]] satellite has a 10-day revisit, but with both satellites (2A + 2B) combined, it's ~5 days. High revisit is critical for time-series and change detection work.
	- Compare with "Orbital period"
- "==[[Swath]] Width==": The width of the strip of ground a satellite images on each pass. A wide swath means more area is covered per pass, but often at lower resolution.
- "==Ground Track==": The path the satellite traces over Earth's surface
- "==Ground Sample Distance==" ([[Ground Sample Distance|GSD]]): The most precise term for what people loosely call "resolution"; The real-world size represented by one pixel. WorldView-3's 0.31m GSD means each pixel covers a 31cm by 31cm square on the ground.
	- GSD varies slightly with off-nadir angles; the further the satellite tilts from straight down, the larger the effective GSD.
- [[Nadir]] vs [[Nadir|Off-Nadir]]:
	- Nadir is the the point directly below the satellite; a straight-down look angle.
	- Off-nadir is when the satellite tilts to image a target to the "side"; taskable satellites can tilt significantly (WorldView-3 up to ~45 degrees) to increase revisit flexibility or collect stereo pairs.
		- Results in larger effective GSD, more geometric distortion, and more visible building/terrain lean.
- "==[[Footprint]]=="
	- The geographic area (polygon) covered by a single scene or image acquisition. Also used for a satellite's instantaneous field of view projected onto the ground.
- "==Scene== / ==Strip== / [[Tile]]"
	- Strip: The continuous ribbon of imagery collected during one pass, before being cut into scenes.
	- Scene: A discrete image product, usually with a fixed-size geographic chunk (e.g. 100km x 100km tiles)
	- Tile: Often used interchangeably with scene, or specifically for pre-defined grid cells (e.g. XYZ web tiles, MGRS tiles, H3 cells)
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
- [[Area of Interest]] (AOI)
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
- ==[[Look Angle]]==: The angle between Nadir (the point directly below the satellite) and where the sensor is pointing. Small look angles give better geometry, while large look angles increase distortion and atmosphere path length.
- ==Atmospheric drag==: Even at LEO altitudes, there's still trace atmosphere. Drag slowly decays orbits, requires periodic reboosts using onboard thrusters to maintain altitude. The reason why LEO satellites have shorter lifespans.
- ==J2 perturbation==: The dominant gravitational perturbation from Earth's equatorial bulge, causing orbital plane precession (the same effect that [[Sun-Synchronous Orbit|Sun-Synchronous]] orbits exploit).
- ==Solar radiation pressure==: Photons from the sun exert a tiny but perceptible force on satellites that matters for orbit determination and altitude control.
- ==Station Keeping==: The ongoing process of firing thrusters to correct for orbital perturbations (atmospheric drag, J2 perturbation, solar radiation pressure) and maintain the desired orbit.
	- [[Geostationary Orbit|Geostationary]] satellites need North-South and East-West station keeping.
	- [[Low Earth Orbit|LEO]] satellites mainly need altitude maintenance.
- ==[[Attitude]]==: The orientation of the satellite in space (roll, pitch, yaw). Not position, attitude is which way it's pointing. This is measured and controlled by ==ADCS systems== ([[Attitude Determination and Control System]]), which uses star trackers, sun sensors, magnetometers to determine orientation, and reaction wheels and magnetic torquers to adjust it.
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
	- ==Irradiance==: The power of electromagnetic radiation *incident* on a surface per unit area, in (W/m^2). "What hits the surface"
	- ==Radiance==: Power emitted or reflected from a surface per unit area pr unit solid angel (W/m^2/sr). What the sensor measures.
	- ==[[Albedo]]:== Fraction of incoming *solar radiation* reflected by a surface across *all directions* and wavelengths. Fresh snow: ~0.9, Ocean: ~0.06. A bulk surface property, *not* a directional measurement.
	- ==BRDF== ([[Bidirectional Reflectance Distribution Function]]): How a surface reflects light differently depending on the illumination angle and viewing angle. A forest reflects very differently depending on if you look at it from directly above vs at an oblique angle.
	- ==Emissivity==: How efficiently a surface radiates thermal energy compared to a perfect blackbody. Water is ~0.99, dry sand is ~0.84.
- Satellites are two separate systems:
	- ==Payload==: Equipment used to carry out the mission the satellite was launched for; cameras, radio equipment for an EO mission, for communication, transponders and high-gain antennas.
	- ==Satellite Bus==: Everything else, the structure and all systems needed to operate the satellite and support the payload.
		- Subsystems
			- ==Mechanical Structure==: Needs to be strong and stiff to survive launch, but also light as possible to save on launch cost. Careful analysis and material selection for lightweight aluminum/carbon-fiber-reinforced polymers. honeycombed composite panels provide surface for mounting equipment. ==Outgassing== causes the vacuum of space to cause materials to gradually release gasses trapped within them, which can condense on sensitive instruments. Metals like Aluminum tend not to outgas significantly, but all materials used on the satellite need to be checked for outgassing; a process called ==Bakeout== (heating under vacuum conditions to accelerate outgassing) is used to reduce risk.
			- ==Thermal Control System==: As the esatellite orbits, it's exposed to impressive temperatures: Surfaces facing hte sun can get hot (~100C), with tempraturesplummeting (~-100C) as it enters the earth's shadow. Add the fact that many of the electronics will be dissipating power, locally increasing temperatures, and it becomes clear that controlling satellite temperature is a huge challenge; good thermal control is mission critical! Certain instruments might need to be in a range of temperatures . Batteries are at risk of failure if outside temperature limits. Thermal control systems make use of many technologies and clever engineering.
				- Because there's not heat transfer by convection in the vacuum of space, the only way to exchange thermal energy with the environment is by radiation.
				- Radiators are large surfaces with high emissivity coatings, used to radiate heat from the satellites. 
				- Heat pipes are used to transport thermal energy from hot to cold areas. 
				- Thermostat-controlled electric heaters switch on at low temperatures to make sure certain components don't get too cold.
				- ==Multi-layer insulation blankets== on the satellite help control temperature by reflecting solar radiation when the satellite is in sunlight, and reducing radiative heat losses when the sunlight is in shadow.
				- Special paints and coatings and phase change materials absorb or release thermal energy by undergoing a phase change.
				- Even the attitude control system can change the direction radiators and other surfaces are facing. All of these components need to work in harmony, perfectly balancing absorbing, retaining, and dissipating heat to make sure the temperature of the satellite stays in acceptable ranges.
			- ==On-Board Computer==: Brain that controls and coordinates all satellite functions, including processing data, monitoring health of satellite, and issuing commands to systems. Several printed circuit boards housed in an aluminum enclosure. Exposure to radiation in space can disrupt circuits and lead to system failure; the onboard computer is particularly vulnerable. Although radiation is low in LEO, those that are in higher orbits (those that pass through the [[Van Allen Radiation Belt]]s) need radiation-hardened components and shielded parts.
			- ==Electrical Power System==:  Generates and stores the power needed to operate; a one-square meter  directly facing the sun in LEO will receive 1370W of solar power, so the most common method is the use of solar arrays. Deployable panels are used to maximize generated power, while allowing the satellite to fit in the fairing of the launcher. Often articulated arrays that can be pointed in the direction of the sun. ==Triple-junction solar cells== (35%) are used to capture a wider range of wavelengths than less expensive ==single-junction solar cells== (20%). Satellites experience periods of Eclipse when in earth's shadow, when solar rays can't generate power. As a result, satellites carry batteries charged by solar panels during periods of sunlight exposure, and discharged during periods of eclipse. The satellite's power control unit interfaces with the onboard computer and controls this process, regulating voltage to ensure stable power supply to all systems.
			- ==Propulsion System==: C be used to move the satellite to a new orbit, or for station keeping (corrective maneuvers to maintain an existing orbit), but can be used for attitude control too. All use by thrust generated by accelerating mass through a nozzle.
				- ==Cold Gas propulsion==: Controlled expansion of a cold pressurized inert gas like nitrogen through thruster nozzles. Low thrust but simple design and precise control; good choice for attitude control systems.
				- ==Chemical propulsion==: Can generate higher thrust, using controlled chemical reactions. 
					- ==Monopropoellant system==: Propellant liquids that compose into hot gasses when brought into contact with a catalyst gas (pressurant).  Lower thrust, but simpler system design.
					- ==Bipropellant system==: Uses two propellants , a fuel and an oxidized, which are mixed and ignited, producing exhaust gasses.  Complex, but offer high thrust, useful for large satellites.
				- ==Electric propulsion==: Uses electric energy to generate thrust, ionizing a propellant line Xenon, and then using electric and magnetic fields to accelerate the ions to high velocities, before expelling them through a nozzle. Lower thrust than chemical propulsion, but is very efficient, requiring less propellant.
			- ==Attitude Determination and Control System== ([[Attitude Determination and Control System|ADCS]]): Used to determine and adjust how the satellite is oriented relative to a reference frame. This orientation is the satellite [[Attitude]], which is constantly being adjusted, and crucial immediately after separation from a launch vehicle, when they may experience tumbling. Also important for normal operation; a satellite might need to point the payload at a specific location, point solar panels to face the sun, and point an antenna towards a ground station. Made up of sensors (star trackers, inertial measurement units, sun sensors) used to determine the current satellite attitude, and actuators (reaction wheels, magnetorquers) used to make the adjustments.  Selected based on the requirements of the mission. A satellite with fewer pointing requirements might use IMUs and Sun Sensors.
				- Attitude determination
					- [[Inertial Measurement Unit|IMU]]: Contains gyroscopes and accelerometers; the three gyroscopes provide changes in satellite orientation, but gyroscopes only provide a relative measurement, and can drift over time, resulting in an error in predicted attitude. Commonly we determine attitude by combining attitude measurement with dat from star trackers
					- ==Star Trackers==: Provide highly accurate  reference measurements of attitude that are used to periodically correct measurements from gyroscopes. Might operate anywhere from every 10s to every 20 minutes. A camera that captures images in the sky, and uses an algorithm to identify bright stars. They use an algorithm to identify bright stars; their position in the image is compared with a catalogue of known stars, letting the position of the satellite to be determined. Need to have clear FoV and can be affected by thermal distortion.
					- ==Sun Sensors==: Simple devices that use photo detector cells to estimate hte attitude of the satellite relative to the sun. Don't work during the eclipse portion of the orbit.
					- ==Magnetometers==: Measure the strength and direction of the magnetic field experienced by the satellite, compared with the model of the earth's magnetic field, and then used to estimate the satellite's attitude, give its position.
				- Attitude Control
					- Most modern satellites use three-axis stabilization, an approach to attitude control where actuators are used to precisely control satellite orientation around three axes.
					- ==Reaction Wheel==: which consists of a flywheel attached to an electric motor. Using the motor to change the rotational velocity of the reaction wheel cause the satellite to rotate around its center of mass in the opposite direction. Three reaction wheels in orthogonal planes gives the satellite the ability to control in three dimensions.  
						- After a certain number of attitude adjustments, the incremental increases in the rotational velocity of the flywheel can bring it close to its max allowable rotation speed (==saturation==). This means that reaction wheels can't be used as the only method for attitude control!
					- ==Magnetorquers==: Used to de-saturate reaction wheels and control attitude. A coil of conductive wire wound around a magnetic core. Passing current through the coil creates a magnetic field. When the field interacts with the earth's magnetic field, a torque is generated. This can be used to control the satellite's attitude, or can be used to slow down the wheels.
			- ==Communications System==: Two separate capabilities:
				- ==Downlink==: Used to beam the data generated by the payload down to earth. Uses the Attitude control system to point its downlink antenna at a groundstation and starts transmitting data. The way this is done depends:
					- Some transmit data once per orbit for each pass over the same groundstation
					- Others transmit to same groundstations (those in geostationary orbit can use the same one continuously).
					- They transmit to groundstations using EM waves, characterized by their wavelength and frequency. Visible light, X-Rays, and Gamma rays are all just EM waves with different frequencies. Satellite communications use waves in the Radio Frequency (RF) part of the spectrum, mostly between 1Ghz and 40Ghz, which is split into bands with designated names:
						- L, S, C, X, K_U, K, K_A bands. As the frequency increases, the required power increases, but the data rates are also higher, and higher frequencies are more susceptible to degradation by atmospheric [[Attenuation]].
						- The data transmitted from a satellite is essentially a stream of 0,1 bits, called the ==data stream==. To transmit this over vast distances, the digital data stream needs to be encoded onto an analog  ==carrier signal==, a continuous sinusoidal electromagnetic wave in a frequency in one of the bands above.
						- This is done by changing certain properties of the carrier signal in a process called ==modulation==:
							- ==Amplitude modulation==: Changes the amplitude of the carrier wave
							- ==Frequency modulation==: Changes the frequency of the carrier wave instead.
							- ==Phase Shift Keying==: Adjusts the phase of the carrier wave.
								- Most high-rate communication links use a form of this, providing high data throughput and low error rates.
									- Binary Phase-Shift Keying: Uses 2 phases
									- Quadrature Phase-Shift Keying: Uses 4 phases
									- 8 Phase-Shift Keying: Uses 8 phases
					- Modulated signal is generated on the satellite by the  transmitter hardware, then through a series of switches and filters before arriving at the downlink antenna for transmission to the groundstation.
					- At the groundstation, the signal is demodulated and processed to obtain valuable payload data.
				- ==TT&C==: ==Telemetry, Tracking, and Command System==: Consists of a transceivers, various filters, switches, and antennas.
					- Has three functions:
						- ==Command function==: Lets operations team control the satellite using commands transmitted from the groundstation, demodulated from the TT&C hardware on the satellite, then routed to the onboard computer for implementation.
						- ==Telemetry function==: Transmits housekeeping data from various sensors on the satellite down to the groundstation, like the temperature of critical components, the power levels in the batteries, or propellant levels in the tanks.
						- ==Tracking function==: Provides information about the position and speed of the satellite. Ground station sends a signal to the satellite, which the transceiver receive and sends back. The turnaround time gives an estimate of the distance between the satellite and the groundstation, and the [[Doppler Effect]] frequency shift provides an estimate of the velocity of the satellite. These are used for the operations team to monitor the position and trajectory of the satellite. Many satellites also carry GPS receivers to enhance tracking capabilities.
				- Satellite onboard antennas come in many different shapes and sizes (Helical, Patch, Reflector, Horn, Wire).
					- It's useful to think about a theoretical antenna called an ==Isotropic Antenna==, which radiates signal in all directions. Two points an equal distance from the antenna will receives the same signal.
					- Usually this isn't what we want: Real antennas are engineered to focus or receive energy in specific directions. This directionality can be visualized as a radiation pattern around the antenna, and is quantified by a parameter called ==Gain==, which is a measure of how much an antenna focuses energy in a specific direction, relative to an Isotropic antenna (gain of 1.0 in all directions). A real antenna will have one that's greater than 1 in some directions, and less than 1 in other directions.
					- ==Antennas are selected to optimize performance for a specific task==
						- Downlink use high-gain antennas that focus the signal into a tight beam, enabling high data transfer rates. 
							- Antenna pointing mechanisms or attitude control system can be used to direct this beam to groundstations.
						- TT&C prioritize reliable communications, instead of high data rate transfers, using low-gain antennas with wider coverage, to make sure that the satellite can communicate with the ground station in all conditions, even if it's tumbling or if the attitude control system isn't' working properly.
- Satellite Mass Classification
	- < 1kg: Pico Satellites
	- < 10kg: Nano
		- The [[CubeSat]]! Developed around the concept of a standardized cube-shaped unit, called 1U.
		- Designed to be modular and low cost; can scale up to 3U, 6U, and 12U being common sizes.
	- < 100kg: Micro
	- < 500kg: Mini
	- < 1000kg: Medium
	- > 1000kg: Large




## Satellite Operational Modes: Tasked vs Systematic Acquisition
1. ==Systematic==: Satellite collects everything in its swath continuously, regardless of specific requests. [[Sentinel|Sentinel-1]] and [[Sentinel|Sentinel-2]] operate this way, they just image everything on a fixed schedule.
2. ==Tasked==: A customer or operator directs the satellite to a specific target. This is how [[Maxar]]'s [[Maxar|WorldView]] operates: someone pays to point the satellite at a location (finite resource, competing requests, priority queueing).



![[Pasted image 20260417124944.png]]


![[Pasted image 20260421105630.png]]





![[Pasted image 20260415115045.png]]
Above: [Source](https://nanoavionics.com/blog/how-many-satellites-are-in-space/)

![[Pasted image 20260415115109.png]]
Above: [Source](https://nanoavionics.com/blog/how-many-satellites-are-in-space/)
- By the 2020s, small satellites comprised 94% of all satellite launches.