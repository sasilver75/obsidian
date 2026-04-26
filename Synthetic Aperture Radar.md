---
aliases:
  - SAR
  - L-Band
  - S-Band
  - X-Band
  - C-Band
  - P-Band
  - HH
  - HV
  - VH
  - VV
  - Co-Pol
  - Cross-Pol
  - Stripmap
  - ScanSAR
  - Spotlight
  - TOPSAR
  - Quad-Pol
  - Dual-Pol
---
Resources:
- [NASA EarthData: Synthetic Aperture Radar](https://www.earthdata.nasa.gov/learn/earth-observation-data-basics/sar) (Unread, looks great)
- [Video: ICEEYE: The Value of SAR](https://youtu.be/GwyldSNlefk) (Awesome, canonical; Tom Ager... WATCH THE FOLLOWUPS ON ICEYE SITE!)
- [Video: GEOINT 2023, a SAR Discussion with Tom Ager](https://youtu.be/4IEp1L7re1A) (Good details)
- Video: [Max Lenormand: How Radar Satellites See Through Clouds](https://youtu.be/zMsCyEAOrh0) (Incredible)
- Video: [Minds behind Maps: Breaking down Radar Satellite Images: Tom Ager](https://youtu.be/qEtg9VTfNj4)

==A [[Remote Sensing]] technique that uses radar signals (microwave bands) instead of visible light to image the Earth's surface.==
- ==Can see at night, can see through clouds.==
	- ==Approximately 67% to 70% of Earth's surface is covered by clouds at any given time==
- The heritage of SAR is that it was usually used for intelligence, or very large civil programs (e.g. NASA, CSA, ESA).
- Just over a decade ago, a few very capable SAR satellites [[TerraSAR-X]], [[RADARSAT|RADARSAT-2]], [[COSMO-SkyMed]]. These are public/private partnerships and massive satellites, over 1,000kg. 
	- [[ICEYE]] revolution is that they have a small SAR satellite constellation funded primarily by VC, with mass of about 100kg and much cheaper, so there can be more.

![[Pasted image 20260425183333.png|1963]]

"Aperture", like in photography, is the size of our lens. It's the thing that captures signal. In our case, it's our (small) antenna, which we're synthetically lengthening by taking multiple observations along the arc of our orbit.
- Like how long exposure photography reveals more stars, the more we look with our radar, the more resolution/details we can see.
- So our resolution is completely separated from how far our satellite is from the ground!

Another tricky thing:
- The pixels in our images aren't necessarily square. In an optical image, the pictures are square, but in a radar image... that isn't necessarily the case. The two dimensions are independent:
	- Range resolution: How close can two objects be for the pulse tot ell them apart; this depends on how short of a pulse we can create. These pulses require a lot of power, more than can be fit on a solar-powered satellite.
		- To lower this below a meter, the trick was we had to modulate the frequency of the pulse. This let us see much finer without having to have a lot more power.
	- Angular resolution: How narrow can we make our beams? That's determined by how narrow of an antenna we can build.


Signal to Noise is the challenge for space-borne SAR:
- When SAR sends a radar pulse (thousands per second), it leaves the antenna... for ICEYE, that pulse is 3000W of power, a strong pulse. It goes to the ground and starts to widen, like inflating a *balloon*. When you inflate a balloon, the skin gets thinner, so the *power density of the pulse weakens with the square of the distance*, as it has to go like 700km to the ground.
- It hits the ground, and a portion of it reflects, and as it comes back, it weakens again with the square of the distance, coming back to a small antenna that can only capture a small amount of that backscatter/return.
- When it hits the antenna, what would you guess is the strength of it (in Watts) when it hits the antenna? Would you say... 100W? 10W? It's 10^-17 Watts! 0.00000000000000001, or one hundred-quadrillionth of a Watt.
- When Tom Ager first saw this, he was astounded! Electric engineers said "Yeah, how did you think it works?"
- This thing that comes back is so enfeebled that any *noise* is a challenge for it!
	- Noise is any microwave hitting the antenna that isn't our pulsed signal.
	- Where would other microwaves be coming from?
		- One source is called the cosmic background microwave radiation from the big bang! Those are microwaves traveling across the universe for 13 billion years, and some of them smack into the antenna.
		- So some of the dots that you see in the SAR image are signals from the big bang, which is wonderful! 
		- The system can't distinguish them from the "real" pulsed microwaves, they're the same.
		- Fundamental Principle: All objects above absolute zero emit electromagnetic radiation
			- So the recorder on the sensor/receiver also operates at a temperature, so IT emits microwaves!
			- "The Poignant Story of Signals and Noise": The receiver itself creates microwaves that confuses it, that it thinks are the reflected signal! We can't stop it unless we can put up receivers that operate at absolute zero, which isn't going to happen. This is called Thermal Noise.
	- So noise will always occur!
		- You need a strong pulse (3000W) and a big antenna (but SAR satellites don't have big antennas!). So that's one of the things you give up when you have little SAR satellites like ICEYE's. One thing you don't give up, though is resolution.

Why don't you give up resolution with small-antenna'd SAR?
- The flight direction is called the Azimuth direction. The longer you image in that direction, the better the azimuth resolution. In spotlight mode, we're targeting one position on the ground.
- The resolution in the Range direction is a function of how much we chirp the pulse; We send a pulse out, but we vary the pulse, like a bird's chirp! That's 300Mhz or so, which gives you on the ground roughly a meter. Soon (from 2023) it will be 600 Mhz, which will be half-meter, then it will be 1200Mhz, quarter-meter on the ground.
- The resolution of the little baby SARs will be those numbers within two years (about 2025); and it won't really get better than that even 10-20 years from now, because once you make the aperture length that long (eg 200km) imposes collection and processing burdens that are difficult to deal with. 
	- The range resolution is limited by the international telecommunications union, which says that you can't have a pulse bandwidth bigger than 1200 Mhz because it interferes with other signals, like 5G. So we'll be sitting around a quarter-meter resolution in range resolution.

Use Cases for SAR:
- People love resolution and will be asking for high resolution (of course, we can do high coverage to do things like look for ships at sea).
- Used by the Ukranians to see Russian mechanized equipment, etc. Learning how to use SAR, talking to the Ukranians might be a good place to start.


A satellite emits microwave pulses towards the ground and records the energy that bounces back; as the satellite moves forwards along its orbit, it takes many measurements from slightly different positions.

The *synthetic aperture* part refers to the fact that you mathematically combine these measurements, as if we had a single enormous antenna that stretched along the flight path, to produce a much sharper image than a real antenna of that size could achieve.
- As the satellite moves along its orbit, it repeatedly emits and receives pulses at the same target from slightly different positions; the onboard computer combines all of these observations mathematically, as if they can from a single enormous "synthesized" antenna.

Why it matters:
- ==Sees through clouds and darkness==: Radar doesn't depend on sunlight, and penetrates cloud cover.
	- Optical satellites like [[Sentinel|Sentinel-2]] and [[Landsat]] are useless over persistent cloud cover, while SAR still has value.
- ==Detects surface roughness and moisture==: The backscatter signal encodes how rough, wet, or metallic a surface is, not just its color. A ship on calm water appears as a very bright spot against a dark background, which is why it's useful for vessel detection.
- ==Sensitive to change==: Comparing two SAR passes of the same area ([[InSAR]]) lets you detect millimeter-scale ground deformation, which is used for earthquake/volcano monitoring and subsidence tracking.
- ==Detects structure, not color==: The signal encodes surface roughness, geometry, and dielectric properties (water content, metal).
	- ==Tradeoff:== SAR imagery looks nothing like a photograph; it takes practice to interpret, and artifacts like speckle noise require specific preprocessing before ML models can use it!



# SAR Bands
- In optical, "bands" mean wavelength channels (red, [[Near Infrared|NIR]], [[Short-Wave Infrared|SWIR]]). In SAR, "bands" refers to the microwave frequency that the radar transmits, and the data you get back is about how the surface scatters that signal, not reflected sunlight.
- ==Key Intuition:== ==Wavelength ~= penetration depth==
	- Shorter wavelength: bounces off surfaces, good for texture, structure, surface motion
	- Longer wavelength: Penetrates through, good for what's underneath (canopy, soil, ice)
- ==X-Band== Radar: ~3cm wavelength, 8-12GHz, best for fine detail (urban, sea ice surface)
	- [[TerraSAR-X]], Cosmo-SkyMed, [[Sentinel|Sentinel-1]] (partly)
- ==C-Band== Radar: ~5.6cm wavelength, 4-8GHz, best for surface change, floods, agriculture
	- [[Sentinel|Sentinel-1]], [[RADARSAT|RADARSAT-2]]
- ==S-Band== Radar: ~10cm wavelength, 2-4GHz, best for (transitional), soil moisture
	- [[NISAR]]
- ==L-Band== Radar: ~24cm wavelength, 1-2GHz, best for forest biomass, soil, subsurface
	- [[PALSAR|PALSAR-2]], [[NISAR]]
- ==P-Band== Radar: ~70cm wavelength, 0.3-1GHz, best for deep forest biomass, root zone
	- Key sensors: BIOMASS (ESA, 2024)

# SAR Polarization
- SAR has various polarization channels:
	- ==HH==: Transmit horizontal, receive horizontal
	- ==VV==: Transmit vertical, receive vertical
	- ==HV==: Transmit horizontal, receive vertical (cross-pol)
	- ==VH==: Transmit vertical, receive horizontal (cross-pol)
- ==Co-pol== (HH, VV): Surface and double-bounce scattering (ground, buildings, water)
- ==Cross-pol== (HV, VH): Volume Scattering (vegetation canopy, rough surfaces)

# Scattering Mechanisms
Decomposing a quad-pol image into these three mechanisms (Pauli, Freeman-Durden decomposition) is a core SAR analysis technique
- ==Surface scattering==: Flat/rough bare surfaces, strong in VV, bare soil, calm water
- ==Double-bounce==: Two perpendicular surfaces (wall, ground), HH, urban, flooded trees
- ==Volume scattering==: Many random scatters (leaves, branches), HV, dense vegetation

# [[Interferometry]] ([[InSAR]]): The Phase Dimension
- Beyond intensity, SAR records ==phase==, the position in the wave cycle when backscatter is returned.
- Two passes over the same area let you compute phase difference, which can help you deduce surface deformation (change) at millimeter-level precision.

Common SAR satellites:
- [[Sentinel|Sentinel-1]] ([[European Space Agency|ESA]])
- [[RADARSAT]] (Canada)
- [[TerraSAR-X]] (Commercial, high-res)
- [[PALSAR]] (Japanese, L-band)

![[Pasted image 20260419191046.png]]
Above:
- Four SAR imaging modes
	- ==Stripmap==: The default, simplest mode. Antenna points at a fixed angle off to the side, satellite flies forward, sensor illuminates a continuous strip. Like dragging a flashlight beam along the ground at a fixed width.
	- ==ScanSAR==: Antenna electronically sweeps across multiple adjacent strips (subswaths) during the pass, trading resolution for width. You cover more ground per pass, but each subswath gets illuminated for less time, so fewer radar pulses hit each point, resulting in coarser resolution. This is how [[Sentinel|Sentinel-1]]'s wide-area mode works.
	- ==Spotlight==: The antenna rotates to keep dwelling on one target as the satellite passes overhead. That one patch gets illuminated for much longer, many more pulses result in finer resolution, but you can only do this for one small area at a time, sacrificing swath entirely. Used when you need maximum detail on a specific point of interest.
	- ==TOPSAR:== A hybrid that fixes ScanSAR's main problem. Standard ScanSAR has *scalloping*, intensity variations between subswathes because edge areas get fewer pulses than center areas. TOPSAR sweeps the antenna in *bursts* with controlled timing, spreading illumination more evenly across each subswath. The result is a wide swath like ScanSAR but without the scalloping artifacts, and compatible with [[InSAR]], which standard ScanSAR isn't!
- Core tradeoff: ==The longer you illuminate a patch, the finer your resolution, but dwelling on one patch means you're not illuminating anything else!==


![[Pasted image 20260420213305.png]]
Above: Showing [[Synthetic Aperture Radar|L-Band]] vs [[Synthetic Aperture Radar|C-Band]] vs [[Synthetic Aperture Radar|X-Band]]


__________

[ICEYE: The Value of SAR](https://www.youtube.com/watch?v=GwyldSNlefk)

We're going to send discrete pulses of radar to the gorund:
- We measure the range to the ground and the strength of the echo
- When the radar pulse hits the ground, it reflects in many directions, which is the backscatter. We capture the backscatter, and the intensity of the received backscatter is the brightness of the pixel.
- The ==Pulse Repetition Frequency (PRF)== is the rate at which we send these pulses. For an aircraft it's many hundreds a second, for a satellite, it might be 2-3,000 pulses per second.
- For optical cameras, looking directly down at the ground gives you the best resolution. For SAR, that doesn't work. SAR measures *range*. If we illuminated the ground vertically, we could have two points with exactly the same range in different places. Those points for a given line would be placed at the same pixel location on the image, and it wouldn't work.
	- ![[Pasted image 20260420221421.png]]
	- So for SAR, we have to be side-looking. In this configuration, every point on the ground has a discrete range, and therefore all the pixels would be in different locations based on their range. 
	- ![[Pasted image 20260420221431.png]]


Most SARs illuminate *broadside,* meaning 90 degrees to the flight path. Some sensors can squint, meaning illuminate the ground forward or behind the broadside. This is called squinting, and the angel is the ==squint angle==.
![[Pasted image 20260420221645.png]]

The ==grazing angle== is the angle between the line of sight ray and the tangent to the ground.

The ==incidence angle== is the angle between the line of sight ray and the local vertical.

Different people use these two measures: Some say incidence angle, some say grazing angle. So for the vague "look angle," you have to understand which they're talking about.

Notice that the two of them form a right angle, so a 30 degree grazing angle is equivalent ot a 60 degree incidence angle.


![[Pasted image 20260421002027.png]]

The illuminatino to the ground is the Slant Range.
![[Pasted image 20260421002147.png]]
The dimension along the flight path is the ==Azimuth range==.
The range along the ground surface is called the ==Ground range==.
![[Pasted image 20260421002203.png]]
So we have radar dimensions in ==Azimuth Range== and ==Ground Range==.

The two resolutions are:
![[Pasted image 20260421002219.png]]
==Azimuth Resolution== and ==Range Resolution==. The ground range resolution is what really matters, because we build our images on the ground...


So why do we call it synthetic aperture radar?
We use ==Coherent Illumination== in SAR; The photons emitted from the sun consist of many different wavelengths of light (blue, green, ultraviolet, infrared), and those photons are all offset from eachother, and the illumination is chaos:
![[Pasted image 20260421002338.png]]
In contrast, the illumination from a Radar or from a laser is ==coherent==; that is, they only send out one wavelength of light (e.g. red for a laser, or 3cm microwave illumination from a radar, like [[Synthetic Aperture Radar|X-Band]] for ICEYE).
![[Pasted image 20260421002420.png]]
They're always in phase with eachother, up and down, like synchronized swimmers.
So ==Coherent Illumination== is controlled and consistent illumination of a single wavelength. This enables SAR imaging.

Because it's coherent, we can measure the ==Phase== of our return, the state of the wave when it comes back to the antenna. 
- Is it at its peak, its trough, or somewhere in between?
![[Pasted image 20260421002512.png]]

We use a small antenna to send our pulses and measure range, brightness, phase for every echo.
- As we move in the flight direction:
	- Send a pulse
	- Measure the echo
	- Move a little bit
- ...and as we do that, we measure:
	- Brightness
	- Phase
	- Range
And in the end, we have an accumulation of data of all of the echoes accumulated in our flight to form images on the ground using sophisticated software.
- Our ground processors use the radar measurements to build images.

==The Synthetic Aperture is the length that the antenna moved during the illumination period==
- If I'm sending 2k pulses a second, the synthetic aperture is the length the antenna traveled during the illumination time.
- We use a small antenna (3m) to create a synthetic aperture by accumulating echo measures from discrete locations.
- ==Azimuth resolution improves as the synthetic aperture gets longer.==

The other dimension of a SAR image is the range dimension, and the resolution in the range dimension is a function of the structure of the pulse we send to the ground.
- We ==chirp== the pulse, meaning we vary the frequency of the pulse in its length!
![[Pasted image 20260421002831.png]]
- The frequency of he pulse is changing, it's not just a single frequency, it changes during the pulse length.
	- If you do this to a sound wave, it soundsl ike a bird's chirp.
- In SAR illumination, this allows us to have very high Range Resolution.
- The Pulse Bandwidth is the amount we change the frequency of the pulse during its length. 
	- Gen3 satellites at [[ICEYE]] have a 600Mhz chirped pulse, meaning the frequency changes by 600Mhz during the illumination time.
	- 300Mhz pulses are common, which give you about 0.5 meter resolution in the slant  range direction, or a bout a meter on the ground.
	- 600Mhz give you about a quarter meter in the slant range direction.
- It's important to realize that ==Azimuth Resolution and Range Resolution are independent of eachother==.
	- Azimuth Resolution : Function of the synthetic aperture length
	- Range Resolution: Function of how much we chirp the pulse when we send it down in the range direction.
- Both are capable of quarter meter resolution, and both (==amazingly==) are ==independent of distance==!
	- With optical, as you move your camera away, resolution degrades. ==With SAR, that doesn't happen==!
		- In the 1960s, this was actually a ==SECRET FACT==!
		- NOTE: The resolution doesn't suffer, but the signal to noise ratio does suffer. Anyone with a 100MP camera knows that just because a sensor has high resolution, doesn't mean it always takes good pictures, especially not at night. While there are many pixel sensors, each one receives so little light that hte data just gets noisy. ==The same thing happens with radar satellites; the resolution might not change iwth distnace, but the quality of the image degrades==. It turns out that we can push this far: We've bounced pulses from earth to the moon to map its surface at 70cm resolution, and the moon is 385,000 km away! 
- SAR though is sensitive to distance, because as it moves away from the ground, the signal strength weakens tremendously, so SAR antennas have to be very sensitive to measure the weak reflections. But resolution itself does not degrade with distance.
- Not too long from now, ==SAR satellites will achieve the practical maximum resolution in both Azimuth Resolution and Range Resolution==.

We can produce two types of images:
![[Pasted image 20260421003354.png]]
Used for viewing, have brightness information: ==Amplitude Image==

Phase image
![[Pasted image 20260421003403.png]]
Which has the phase for every pixel: ==Phase Image==. Valuable!
- This may seem random and chaotic, but it is not, and it's very useful. We can create some special products that we'll talk about in our later talks. 

We can combine the two into a ==Complex Image==
- SAR engineers use complex numbers (which have two values) to store this information; every pixel has an amplitude and a phase value.
- ==The beating heart of SAR is its coherency, it's phase data. People often throw it away!== Understand that a real SAR image is a complex image!
	- You can make a useful product from Phase called a ==[[Color Subaperture Image]] (CSI)==; if you're collecting Azimuth for a long time, getting high Azimuth resolution... imagine that instead of keeping the entire synthetic aperture, you just processed the first half of it. You could still make an image, with Azimuth resolution of 1/2. Then you can make one with the second half, and overlay them. You can do that with 3 of them (1/3 each), and make the first red, the second green, and the third blue. If the reflections are the same and you overlap them, you see the usual grayscale image you're used to. But suppose that in the first image... there's a Russian mechanized troop transport, which in the first part of the aperture *glints*. On the image, it shows up red. SAR reflects off things with straight lines and cubic structures: things that human built, not nature. So this color subaperture image shows colors for glints, and those tend to be HUMAN-MADE objects!
	- We need to make amplitude and phase easily exploitable. 
	- You can output the complex image in amplitude and phase, and put it in [[GeoTIFF]] or [[Cloud-Optimized GeoTIFF|COG]]. 

"The great, brilliant Radar people I once new are getting older and retiring, and I don't believe they're getting replaced."

Understand: SAR produces a complex image with amplitude and phase information.
- If I send you an image for viewing, you'll get the Amplitude pixels only. All of the valuable Phase information is thrown away and it's not available to you!
- Soon, we'l have to find a way to make Phase information available to regular users, and not just useful for scientists and engineers.

Einstein's gravity waves were measured by phase changes.  That's how SAR works; ==it measures little phase changes... which are measure with precision much much less than a single wavelength.==

We talk about SAR resolution... we usually think about spatial resolution (Azimuth Resolution, ==Range Resolution==; one based on range of synthetic aperture, and the other on Chirp), but there's also ==Phase Resolution==, which is much much less than a single 
wavelength.

We have a number of different imaging modes with SAR:
![[Pasted image 20260421003742.png]]
![[Pasted image 20260421003751.png]]
==Spot== is the highest resolution, we vary our illumination angle a little bit to image a single area on the ground as we move along, and it's like I shined a spotlight at the single place on the ground.
- We get a very long synthetic aperture over a very small area.
- We have a ==3m antenna==, but I'm going to illuminate the ground for ==10-20 seconds== (@ ~7.5km/s in [[Low Earth Orbit|LEO]]), which produces a synthetic aperture (antenna) that's ==over 100km long,,== using a tiny 3m antenna! It's as if we have an antenna in space THAT long, and that's an astonishing capability.
- ![[Pasted image 20260421004232.png]]
- Helsinki in Spot mode


In ==Strip== mode, we set our illumination to shoot broadside, and shoot, accumulating our image in this way, building it up by sending our pulses... not changing the illumination direction as we do in spot mode.
![[Pasted image 20260421004055.png]]
This gives a much larger image at reduced resolution; maybe 3 meter resolution, instead of less than a meter in spotlight mode.
![[Pasted image 20260421004143.png]]
San Francisco, California in Strip mode: See how well you can see the texture in the water in SAR?




In ==Scan== mode:
![[Pasted image 20260421004407.png]]
![[Pasted image 20260421004355.png]]
We get modest resolution, but  very wide swaths by changing the incidence angle.
- By changing the incidence angle and sweeping the area on the ground a little bit, we get multiple overlapping strips.
- The cost is reduced Azimuth Resolution, but we get very wide areas / long strips in scan mode.
![[Pasted image 20260421004502.png]]
Example Scan mode; This was taken shortly after the large cargo container was freed that was blocking the Suez Canal (Ever Given). It was 100km long, and extremely wide. 
- ![[Pasted image 20260421004621.png]] 
- See how nicely SAR shows topography?
- See the ships waiting to get into the Suez?
- Se the lit up ships in the canal?
All on a single image taken from a tiny SAR satellite.
- Scan mode wasn't planned when these satellites were designed, and became operation via a simple software upload to the satellite! Tremendous innovation.

(Aside: And it seem they even have a new mode, ==Dwell Mode==, since this talk)
![[Pasted image 20260421013926.png]]
- In Dwell Mode. they're imaging a spot on the ground for 25 seconds... and if it flies at 7.5km/s for 25 seconds, that's almost 190km during the collection, so the synthetic aperture length is 190km. It's as if you have an antenna in space 190km long; the resolution for that collection... is 5 centimeters.


We know that SAR penetrates clouds and darkness.
- It uses ==Coherent Illumination,== producing amplitude and phase information; phase information is useful for advanced analysis.
- Resolution for SAR is independent of distance!
- Multiple imaging modes; Spot, Strip, Scan modes... soon, it will be at its ***practical maximum resolution***, but less than a a meter in both directions.
	- ((Wonder what he means by practical maximum resolution))


___________________

Video: [Minds behind Maps: Breaking down Radar Satellite Images: Tom Ager](https://youtu.be/qEtg9VTfNj4)

- 60-70% of the Earth is covered by clouds; SAR is very important.
- 














