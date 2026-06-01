---
aliases:
  - Radio Detection and Ranging
---
Resources:
- Video: [Max Lenormand: How Radar Satellites See through Clouds](https://youtu.be/zMsCyEAOrh0) (This is EXCELLENT, explaining use of RADAR in [[Synthetic Aperture Radar|SAR]])
- Article:[ "An Introduction to Synthetic Aperture Radar Imaging" by Tom Ager in Oceanography magazine](https://tos.org/oceanography/assets/docs/26-2_ager.pdf) (Excellent pictures!)


RADAR's ability to determine range and motion make it suitable for many applications, but its most important attribute for imaging applications is that its relatively long wavelengths penetrate clouds, dust, and even volcanic ash, and it can image independent of most weather conditions. Given that we're on an ocean planet with a lot of water vapor continuously condensing into clouds over large regions of Earth's surface, radar is a core remote-sensing technology and it becomes the primary imaging source when cloud cover prevents other means of data collection.
- Funnily, Radar uses ==microwave-band== waves, not radar band.


-----------
Below are notes from Tom Ager's Oceanography magazine, see Resources.

# The value of Radar Imaging

There are a number of established radar bands (e.g. [[S-Band]], [[X-Band]], [[C-Band]]), each with their own frequencies and wavelengths. Radar imaging avoids wavelengths shorter than K-band (1.11-1.67cm, compared to X-Band's 2.5-3.75cm), because these are reflected by water vapor and other atmospheric particles.
- In fact, high-frequency microwave bands are used in [[Doppler Effect|Doppler]] radar weather instruments to detect rain and storms!
The more typical imaging bands are [[X-Band]], [[C-Band]], and some very long wavelength bands.
- The wavelengths of [[L-Band]] and [[Ultra High Frequency]] (UHF; a confusing term that in the context of SAR bands is actually some of the lowest wavelengths) radar are so long that they can be used to penetrate foliage and see through tree canopy in forested areas.


Radar has other notable characteristics that make it useful for remote-sensing applications:
- Because radar provides its own illumination, data can be collected independent of sunlight.
- [[Synthetic Aperture Radar|SAR]] provides high resolution with the remarkable characteristic thats its resolution does not degrade with distance (although at long distances noise becomes an issue because the returned signal is so weak; returned Wattage drops with the square of distance). 
- Because radars don't have a fixed lens like optical satellites do, they're flexible regarding resolution and ground coverage, so a single system can collect data from wide areas at low resolution, medium areas at medium resolution, and small areas at high resolution.
- The measurements radars make are naturally precise, and radar imaging systems can be configured to have outstanding geometric accuracy.
- Finally, radar illumination is ==coherent==. Radar antennas emit pulses of microwave energy in which the characteristics of the wave are controlled and consistent, pulse to pulse. This natural coherence enables the creation of products like [[Digital Elevation Model|DEM]]s and sensitive measurements of Earth's surface changes over time.
- A radar antenna emits individual pulses of microwave radiation. The pulses are sent at a ==pulse repetition frequency (PRF)== in the range of 2,000/sec or more. These pulses are scattered in every direction, and a small portion, called radar backscatter, is returned to the antenna. The radar measures the characteristics of the echoes, including the round-trip ==time== for the pulse to ravel from the antenna to the ground and back to the antenna, the ==strength== of the reflection, and the ==phase== of the return wave. From the time, we can determine the distance from the antenna to the ground; Range is simple to calculate and is merely equal to the speed of light multiplied by the round trip time divided by two.


![[Pasted image 20260421164942.png]]
Above: See the chart that shows the measurement of the strength of the backscatter as a function of time. In this example, the pulse reflects first in the near-range area. Most of the associated reflection from the flat terrain travels *away* from the antenna, and so the measured backscatter is relatively weak. This is followed by stronger backscatter, caused by the texture of the trees, and then a very weak return from the calm river. The reflection of the front side of the hill is very strong, which would result in bright pixels on the radar image.

# Radar Collection Geometry

![[Pasted image 20260421165219.png]]
Above: 
- Figure 3
	- Airborne and spaceborne radar sensors image off to the side of the sensor's flight path, usually at an angle orthogonal to the flight direction.
		- Fig3 shows this as the "Broadside direction"
	- Some radars can squint the collection to image ares in front or behind the broadside angle.
	- The angle down from the horizon is the Depression Angle.
	- The Incidence Angle is formed between the vector that's perpendicular to level at a particular location and the line between that point and the antenna.
	- The Grazing Angle is the complement to the incidence angle (they add to 90).
- Figure 4
	- A collection mode where the radar is imaging broadside and compiling a long image swath by combining many thousands of individual pulses. This creates radar images with both Range and Azimuth directions. 
		- Range refers to the dimension alongside the broadcast angle
		- Azimuth is orthogonal to that direction.

# An overview of Radar Image Characteristics
- The look of radar images is determined by the ==collection geometry==, the manner in which energy strikes and reflects from an object, and the ==reflectance characteristics== of the object.
- In the examples provided, the trees create a diffuse reflection, which would normally result in moderatte backscatter nad gray pixels on the image.
	- Keep in mind that [[Ultra High Frequency|UHF]] radar bands would not reflect from the trees at all, but instead travel through the leaves and reflect from tree trunks and the ground surface. Objects hidden beneath forest canopy would be apparent in such images.
- Water is normally flat with respect to the long microwaves and strongly reflects energy AWAY from the antenna; thus, it foten is very dark on radar images. If the water surface is rougher, as often it will be in the open ocean, reflections from waves will be relatively bright compared to the background.
- Vertical objects like buildings produce strong returns and appear white on radar images.

![[Pasted image 20260421173202.png]]
Above:
- See the distinction between the Potomac and Anacostia Rivers and the land surface; many corner reflections can be seen in the bridges and the buildings of downtown DC.

![[Pasted image 20260421173429.png]]
Above:
- In the radar collection in Fig 7, the microwave energy illuminates the mountain and the areas in front of it and behind it... The front face with smaller local incidence angle reflects strongly and creates bright pixels, while the surrounding areas produce varying shades of gray. In contrast to the dark shadows on optical images, radar shadows are black; they're null areas because there is no reflection, though they may include signal noise and other image artifacts.
	- Radar shadow cannot be manipulated by adjusting the contrast or brightness of an image, like you can for optical images... Sunlight is [[Attenuation|Attenuated]] by the atmosphere and is scattered sufficiently to illuminate even the regions that are (directly) blocked from seeing the sun.
- In Figure 8, the energy is arriving from the top of the scene, and shadows fall off the backside of the buildings.

![[Pasted image 20260421173715.png]]
- Radar is interesting in how elevated objects are projected onto a radar image.
	- Fig 10
		- Radar determines the pixel location of a reflecting object in part based on its range.
		- Notice that the sloped front end of the mountain in Fig 9 has a very small variation in range! This means that the entire front face of the mountain will be compressed on the image into the space of only a few pixels; this is called [[Foreshortening]].
	- Fig 11
		- A mountain illuminated from a steeper angle... in this case, the top of the mountain has the same range as the area to the left of the mountain, ==meaning that the mountain peak and the flat area with the identical range would occupy the same pixel on the image.==
		- The mountain would lean to that area with an effect called [[Layover]]. Layover is common for mountains, and always occurs for towers and buildings. The degree of layover increases with steeper collection angles, and the direction of layover is perpendicular to the flight direction of the sensor.
	- Fig 12
		- A TerraSAR-X radar image; the radar illumination came from the left in this image, and the sensor motion was up toward the top of the image. Layover is perpendicular to the sensor motion, and the building leans to the left. Notice that the signature of the access road to the left of the building in the optical image is ==partially blocked== by the displaced signature of the roof of the Alamo dome!
- Foreshortening and Layover are often discussed as unique radar image characteristics, but optical images display similar effects on [[Oblique]] imagery: Tall objects lean over on optical images due to [[Relief Displacement]], and pixels on the far side of oblique images are compressed in scale. It tends to be more prominent on radar images though, because they must always be taken at an angle; they cannot be imaged at nadir like the optical image on the right in Figure 12.


# Notes on Polarization
- Radar and other forms of [[Electromagnetic Spectrum|Electromagnetic]] radiation consist of electron and magnetic fields that are perpendicular to eachother.
- The polarization of an EM wave refers to the orientation of the electric field.
- ![[Pasted image 20260421174616.png]]
- If an electric field is oriented vertically with respect to the Earth's surface (left, Figure 13), it's "Vertically" polarized.
- Because radars emit coherent radiation, the polarization is controlled, and the sensor can be configured to emit and record waves of a chosen polarization.
- The look of the final image will be different depending on the polarization used to illuminate the ground, and the physical structure of the reflecting objects!
	- ==Tall objects scatter vertical polarized energy more prominently than horizontally polarized energy!==
	- ==Power lines reflect more strongly from horizontal polarization.==
- While polarization of an EM wave can take a range of orientations, imaging radars typically use vertical or horizontal or some combination of the two.
	- A radar might transmit horizontal pulse and record the horizontally oriented part of the backscatter. This is called an [[Synthetic Aperture Radar|HH]] system.
	- When a pulse of a particular polarizaiton strikes an object, a portion of the backscatter energy might be flipped in orientation, so it's posssible to arrange for the transmit and receive cycles to handle different polarizations. (e.g. [[Synthetic Aperture Radar|HV]]), which is called [[Synthetic Aperture Radar|Cross-Pol]]arization.
![[Pasted image 20260421182122.png]]
- Above: This imageo f SF on the lefet has HH polarization from RADARSAT-2. THe middle has HV: notice that the signatures of the two scenes are quite different!
	- ==Cross-Polarization== is often useful in ship detection applications because, under some conditions, it suppresses the signature of teh open ocean while providing bright signatures for ships.
	- On the other hand, HH polarization retains the ocean signature and can sometimes be used to record wake signatures.
	- The different signatures of the oceans' surface are quite evident in these two imagse!
	- ==Such dual-polarized collections (HH, HV) can be used together for improved ship-detection capabilities.==
- Some radars can change polarization from one pulse to the next! That's where the Figure 14 pictures came from: [[Synthetic Aperture Radar|Quad-Pol]] collection! Because the HV and VH images are alive, it actually produces three distinct images.
![[Pasted image 20260421182359.png]]
- This shows them combined into a false-color composite produced by projecting the HH image in red, the HV in green, and the VV in blue. In this way, many ground cover features can be distinguished!

# Aperture Synthesis: The remarkable story of SAR image resolution
- In the early days of radar imaging, it wasn't possible to create images of good resolution from long standoff distances, because the resolution in the azimuth direction was equivalent to the size of the beam pattern on the ground.
- SAR uses the different locations of the sensor as it moves alongside the flight path recording the backscatter echoes, to simulate or synthesize a large antenna from a small one. In the imaging process, the individual transmit and receive cycles are completed from many locations as the sensor moves. These are treated as if they were array elements of a single long antenna strung out over the flight path, and the measurements of time/amplitude/phase are treated as if they were collected simultaneously from one very long antenna.
![[Pasted image 20260421183332.png]]
- Fig 17 shows the radar antenna illuminating hte straight of Gibraltar with thousands of pulses to form one large synthetic aperture.
	- ((In new products, ICEYE might spot a location for 25s moving at 7.5km/s, for a ~190km long synthetic antenna!))
	- This collection ensures that the reflections are recorded from each of the many receive locations, forming a huge aperture.
		- These locations cannot be separated by MORE than one-half the antenna lanegth, which determines the minimal rate at which pulses *must* be emitted, called the ==pulse retitition frequency.==
- Aperture synthesis is based on the coherent nature of radar illumination:
	- The antenna measures the phase of each of the returns with exquisite precision.
	- For any particular ground reflector, the phase of its echo changes at each receive location. 
	- The radar records the "phase history" of echoes, and SAR image formation is referred to as "Phase History Processing."
	- The SAR azimuth resolution turns out to be dependent only on the wavelength of the microwave energy and the angle subtended by the synthetic aperture.
- SAR azimuth resolution  does not degrade when the coherent collection angle is maintained; SAR azimuth resolution can be quite small, with commercial radars capable of ((sub meter)) resolution at ranges approaching 1,000km.
- Spotlight and Stripmap Mode

![[Pasted image 20260421183738.png]]
Above:
- Spotlight mode vs Stripmap mode. 

## A note on Doppler Shift
- The apparent shift in frequency caused by the relative motion between the source of a wave and the listening device.
	- A train whistle that sounds higher as it pulls up to you, and lower after it passes.
- All of the radar echoes are indeed affected by the Doppler shit, but Doppler frequency variations are not needed for aperture synthesis.
- SAR Processing is instead based on the phase variations caused by the different receive locations of the antenna.
	- These variations are greater for longer synthetic apertures because the echoes are recorded in more locations.
	- The doppler shift has nothing to do with it.
- Phase variations don't require motions; they could be recorded via a series of fixed antennas that happen to replicate the sensor receive locations. In this case, there would be no Doppler shift, yet an identical image oculd be formed.

# Improving Range Resolution

![[Pasted image 20260421184034.png]]
- Fig 20:
	- Radar pulse reflecting from two houses; the pulse is emitted for a short duratino, so it has a pulse length equivalent to C times the pulse duration. This long pulse strikes the first house and reflects from it, then continues to the second house. The front end of the pulse begins to reflect while the backend continues towards the house. In this case, the two houses are sufficiently separate in space to generate two distinct reflections.
- Fig 21:
	- In this case, two houses are closer together, the two reflections are mixed, and the radar would only "see" a single object.
- ==Range resolution for a single pulse is one-half the pulse length.== For two objects ot be distinguished, they must be separated in range by at least half the pulse length.
- So pulse duration is an important variable; even for a pulse of one-millionth of a second, the slant range resolution would be 150m. This is simply not feasible to make much smaller.
- Fortunately, a so called ==chirped== pulse can be used to solve this problem.
- ![[Pasted image 20260421184546.png]]
- A chirped pulse is one in which the frequency varies within the pulse. Fig 22 compares a simple pulse of fixed frequency with a chirped pulse in which the frequency is varied in a steady, or linear, rate. 
	- "Chirp" is used because a sound wave of this form sounds like a bird's chirp.
	- Astonishing fact:  It turns out that ==Range resolution depends sole on the frequency variation of hte pulse, and has nothing to do with distance or duration of the pulse.==
- Though resolution doesn't degrade with distance, the returned signal does, exponentially (and more noise, relatively).

![[Pasted image 20260421184614.png]]

SAR doesn't know anything of imagery, it only knows about its raw measurements of time, amplitude, and phase, and then these are manipulated with signal processing algorithms into images and other products.

A phase history data file is not an image product and can't be viewed, but it is processed into a pixel form called a =="complex image."==
- Complex image has two values for each pixel: amplitude and phase.
![[Pasted image 20260421184806.png]]
- The amplitude image is easy to interpret, while the associated phase image appears to include only a random scattering of measurements; this is misleading, because phase measurements are precise, and of great value!
- The technique called persistent scatter interferometry demonstrate the value of phase information: Multiple radar collections of a site are taken over time from nearly the same location in space, and the resultant images are processed to measure very subtle ground surface changes. If the radar image is controlled so that the ground is illuminated multiple times by the same sensor from locations that vary by only a short baseline distance, than the phase signatures of stable ground objects can be compared to calculate changes in distances between the scatterers and the antenna. 
	- ==Phase measurements are so sensitive that inconsistent changes like *leaf motion* cause phase changes!==
	- Vegetated areas that changed to to wind and rain are often "de-correlated"; their phase signatures cannot be matched from image to image; but phase signatures of stable ground objects can be used to determine very subtle changes in surface structure over time!
	- ![[Pasted image 20260421185115.png]]
	- This shows surface subsidence around an oil field in Kuwait, generated from a stack of 16 TerraSAR-X images collected between 2008 and 2011. This shows ground deflation due to oil field depletion! The subsidence is occurring at a rate of about 10mm per year in the red areas. This is pretty remarkable sensitivity, given that the measurements were made by a spacecraft more than 800km distant!

## Space-based SAR Systems
- Particularly suited to ocean applications because satellites have global access and offer repeated collection opportunities.
- RADARSAT, COSMO-SkyMed, TerrsaSAR-X and TanDEM-X all have been launched an have resolution as good as 1m, including wide-swath ScanSAR modes that cover areas hundreds of kilometers wide.
- It's easy to imagine the value of combining SAR ship detection images with coincident AIS data used to characterize a ship's identity, position, and course. Ships not emitting AIS information can be identified in the SAR image for tracking, and, perhaps, physical searching.
	- Oil and surfacants tend to suppress Brag waves and are usually darker on radar images than surrounding water.
	- Radar can penetrate ice on the order of a few wavelengths, creating volume scattering, which provides a brighter signature for ice than surrounding water, as seen in Figure 29.
	- The small surface waves that SAR is so good at detecting can also be used to extract other information not directly apparent in the image itself:
		- Wind speed and direction
		- Detect activities and structures below the ocean surface, such as underwater currents and shoals.
		- Radar has also bee used to measure speed in rivers via a technique called along-track interferometry, using phase data collected by two antennas separated along the line of flight.

![[Pasted image 20260421190306.png]]
![[Pasted image 20260421190312.png]]
![[Pasted image 20260421190323.png]]



____________________
Adam Stewart AI4EO Class

- ==RADAR doesn't actually use radio waves, it uses microwaves.==

- Meteorology: precipitation, wind using [[Doppler Effect|Doppler]] radar
- [[Ground-Penetrating Radar]]: Can fire a small seismic pulse down into the ground and measure the time it takes for that signal to reflect off of material underground, including things like bunkers, ancient buildings, or oil reserves.
- [[Synthetic Aperture Radar]] (SAR): You actually can't measure small changes in a surface with microwaves, since the wavelength of the light is very large... So you can make your radar receiver, your radar antenna, larger. If you make it larger, it can get detail at a great resolution.
	- There's a physical limit to how large you can make a sensor, but you can physically MOVE the sensor and take multiple readings at different locations (several antennas in a large array, grid), or put that radar sensor in a large satellite and fly it around the earth, which synthetically increases the aperture of the sensor.
	- Now we can make very precise measurements of things like elevation ([[Digital Elevation Model|DEM]]), flooding, or even use it for glaciology and peek into the surface level of ice and snow to understand how thick the ice is on a glacier!
- An extension: [[InSAR|Interferometric Synthetic Aperture Radar]] (InSAR)
	- Taking a SAR image... and taking *another* image at a different time! These images are interesting; they aren't just floating point numbers of intensity; because they use an active light source, they can measure the time it takes for the light source to bounce off the ground, reflect/return to the original satellite. This active signal is very nice, you can make your satellite work at night, working 24/7, but also... they pass through clouds. So even if there are clouds overhead, you can see the surface of the earth.
	- With InSAR, we take two images at two different times, capturing both the intensity and phase of light, firing a polarized light source at the ground. So we have a complex number, with both intensity and phase. Comparing the phase of two images at two times, we can get an image like:
	- ![[Pasted image 20260422115223.png]]
	- This is a locations where there was an earthquake; the bulls-eye patterns are instants of subsidence, where the ground lifted or sunk by millimeters.








