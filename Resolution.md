---
aliases:
  - Radiometric Resolution
  - Spatial Resolution
  - Spectral Resolution
  - Temporal Resolution
---


In [[Remote Sensing]], there are four types of resolution to consider for any dataset, each of which plays a role in how data from an instrument can be used:
- ==Radiometric== Resolution: The amount of information in each pixel; the number of bits representing the energy recorded. The higher the radiometric resolution, the more values are available to store information, providing better discrimination between even the slightest differences in energy.
- ==Spatial== Resolution: The size of each pixel within a digital image, and the area on Earth's surface represented by that pixel.
- ==Spectral== Resolution: The ability of an instrument to *discern* finer wavelengths; that is, having more and narrower bands.
	- Many instruments are considered to be [[Multispectral]], meaning they have 3-10 bands.
	- Some instruments have hundreds to even *thousands* of bands, and are considered to be [[Hyperspectral]].
	- The narrower the range of wavelengths for a given band, the finer the spectral resolution. 
- ==Temporal== Resolution: The time that ti takes for a space-based platform to complete an orbit and revisit the same observation area.
	- Depends on the orbit, the instrument's characteristics, and the swath width.
	- Because [[Geostationary Orbit|Geostationary]] platforms match the rate at which Earth rotates, the temporal resolution is much finer.
	- [[Polar Orbit|Polar]] orbiting platforms have a temporal resolution that can vary from 1 day to 16 days.


Q: ==Why not build an instrument with high radiometric, spatial, spectral, and temporal resolution?==
A: It's difficult to combine all of the desirable features into one remote instrument. To acquire observations with high spatial resolution, a narrower swath is required, which required more time between observations of a given area, resulting in lower temporal resolution.
- Understanding the type of data you need for a given study is critical! When researching weather, which is dynamic over time, a high temporal resolution is critical.
- When researching seasonal vegetation changes, a high temporal resolution can be sacrificed for higher spectral or spatial resorlution.



![[Pasted image 20260416001542.png]]
Above: Overview



![[Pasted image 20260416001732.png]]
Above: Advances in Radiometric Resolution, or how sensitive an instrument is to small differences in electromagnetic energy. 


![[Pasted image 20260416002144.png]]
Above: Spatial Resolution, and use cases at different resolutions

![[Pasted image 20260416002506.png]]
Above: Spatial Resolution differences in a [[Landsat]] 8 image of Reykjavik


![[Pasted image 20260416003010.png]]
Above: Spectral Resolution example; The small region of high response in the right corner of the image is in the red portion of the visible spectrum (~700nm), and is due to the presence of 1cm long red bring shrimp in the evaporation pond.
