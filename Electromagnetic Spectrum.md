---
aliases:
  - EM Spectrum
  - EM
  - Electromagnetic
---
-Electromagnetic energy, produced by the vibration of charged particles, travels in the form of waves through the atmosphere and the vacuum of space.

These waves have different wavelengths (the distance from weave crest to wave crest) and frequencies; a ==shorter wavelength means a higher frequency==.

Some, like radio, microwave, and infrared waves, have a longer wavelength.
Others, like ultraviolet, x-rays, and gamma rays have a much shorter wavelength.
Visible light sits in the middle of that range of long to shortwave radiation.

Some waves are absorbed or reflected by atmospheric components like water vapor and carbon dioxide, while other wavelengths allow for unimpeded movement through the atmosphere.
- Microwave energy has wavelengths that can pass through clouds, which is useful for instruments on weather and communications platforms.

==All things on earth reflect, absorb, or transmit energy; the amount of which varies by wavelength.==

Just as your fingerprint is unique to you, ==everything on Earth has a unique spectral fingerprint.== We can use this information to do things like identify different rock and mineral types! The number of spectral bands an instrument has, its [[Resolution|Spectral Resolution]], determines how much differentiation a researcher can identify between materials.

![[Pasted image 20260416105955.png]]
Above: The Electromagnetic spectrum


![[Pasted image 20260422112545.png]]
- Visible spectrum: We can see basically things that are approximately the wavelength of a single-celled organism, but there are many other wavelengths of light...
- Infrared spectrum: You can actually see temperature, as opposed to visible light. Can give you a sunburn.
- Microwave:  Might not want to stick your head in a microwave, but relatively safe.
- Radio: Safe, going through you right now.
- UV: sunburns can get pretty bad
- X-ray: Can see through your body, can cause cancer with regular exposure
- Gamma rays: Most dangerous, can destroy the atoms in our body if there's enough energy. Earth's atmosphere blocks most of these high-frequency ones (not so much of the ultraviolet spectrum).

Everything emits electromagnetic energy. The sun happens to emit a lot of light in the visible wavelengths. It's no accident that humans have evolved to see this particular wavelength of light. This is largely based on the temperature of the sun itself. 
- As a human, you have  a temperature, and you emit light, which happens to be in the infrared spectrum.
- Even pretty cold objects emits light at some wavelength of the spectrum.

Traditional cameras are designed to capture light in the visible spectrum; this is the part that we're most interested in taking an imagine of, as people that see in the visible spectrum ourselves.
- Separately filter and record different wavelengths of light (typically RGB).
	- ![[Pasted image 20260422113252.png]]
- You record three different bands, each of which are a grayscale image (just intensity, say 0-255), and you can sort of look at these and think about which of them are the grayscale. The top-right is blue, because it's bright/intense in the blue parts of the outfit.
	- When you take these images and stack them on top of eachother, the result is a color image!
	- Typically we record RGB, because those are the parts of the visual spectrum that humans are most sensitive to.
	- Can be active or passive sensors (either creating a light source or just using the ambient light)

![[Pasted image 20260415184933.png]]
Above: [[Panchromatic]], [[Near Infrared]] (NIR), [[Short-Wave Infrared]] (SWIR), [[Thermal Infrared]] (TIR), 


![[Pasted image 20260420194636.png]]
Above: Comparison of [[Landsat]] [[Sentinel|Sentinel-2]] re: [[Hyperspectral]] vs [[Multispectral]]
- The wider the box, the broader the frequency for the band.
- Notice that bands coincide with peaks in atmospheric transmission; these parts of the spectrum are where EM is transmitted through the atmosphere; without that, we wouldn't get reflectance at all.
- Notice... the other thing you might notice is that atmospheric transmission is pretty low in the visible (VIS) part of teh spectrum, which limits the amount of information we can catch in that part.

So why do we capture in different bands?
![[Pasted image 20260420194925.png]]
Above: Remote sensing of vegetation; the visible part of red and blue get absorbed by chloroplasts and are used for photosynthesis! The green is reflected, which is why we see plants as green.
- [[Near Infrared]] is completely ignored by the chloroplasts and interacts with the middle part of the leaf, which means we can understand more about cell structure by looking at NIR and beyond
This is an example of why ==seeing beyond the visible is powerful; it allows us to see and measure things that we can't see with our own eyes!==

![[Pasted image 20260420195112.png]]
To allow us to difference between (eg) vegetation, we use what we know about the reflectance characteristics of those land covers, and create curves like this, which are known as [[Spectral Signature]]s, which is like a fingerprint!
- They're likely to be very similar for all healthy vegetation; it allows us to explore vegetation health; deviation from this expected curve means something weird is happening with health of plants.
	- Conifer vs broadleaf trees are likely to be a little bit different.
- In the visual spectrum, see that water does have a spectral signature, and that beyond NIR, all life is ==[[Attenuation|Attenuated]]== or absorbed, and we expect nothing in reflectance. This means that we can't do too much on the water's surface or below.

![[Pasted image 20260420195319.png]]
Above: The intensity of light decreases exponentially with water depth, and both conditions at the surface as well as turbidity below the surface highly impacts it.`


