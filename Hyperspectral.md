---
aliases:
  - Hyperspectral Imaging
  - HSI
---
c.f. [[Multispectral]]

- Hundreds of *contiguous* spectral bands (whereas multispectral is ~3-5 *disjoint* spectral bands).
	- With a multispectral camera, you might have a small bandwidth covering the red portion of the spectrum, another covering the orange, and a gap in the middle that we don't collect/focus on. With a hyperspectral camera, we capture *every part* of the EM spectrum; there are still bars, but they're touching eachother and have very narrow width; it's like Riemann's sum in Calculus.
- Similar to a mass spectrometer, where you're measuring the individual components of the subject that light is reflecting off of, and you can precisely pinpoint the type of surface or material that it's reflecting off of.
- Examples: [[Hyperion]], [[PROBA]], [[PRISMA]], [[EnMap]]
	- Most on order of 30m per pixel or worse; tend to have worse spatial resolution than multispectral.



![[Pasted image 20260422113849.png]]

![[Pasted image 20260420194450.png]]

![[Pasted image 20260422114505.png]]
- Instead of having an image that's height by width, we also have a new color channel for wavelength.
- So we have different components of the spectrum that measure different things!
- If you look at the cross-section of a particular land surface, you get something like a digital fingerprint for that location, where (say) different species of vegetation will have very different fingerprints, different manmade materials will have slightly different digital signatures captured by this hyperspectral images.
- Tradeoff: The more [[Resolution|Spectral Resolution]] you have, this typically results in a lower [[Resolution|Spatial Resolution]]. These bands are very narrow, and there are very few photons of light traveling per particular band, so you need a larger pixel size to accommodate for that, in order to avoid noise.

