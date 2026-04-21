A technique that extracts information by comparing the **phase difference** between two wave signals that traveled slightly different paths.
- *"Interferometry uses the phase of waves as a ruler; small path differences create measurable interference patterns that reveal millimeter-scale changes.*

Core Physics
- Waves have a ==phase== (where they are in their oscillation cycle). If two waves started together but traveled different distances, they arrive slightly out of phase.
- That phase difference is exquisitely sensitive to path length differences.
- We can essentially use these wave physics as a ruler!
- The process of converting ambiguous 0-2π phase cycles into continuous deformation values (the hard math problem in InSAR) is called ==Phase Unwrapping==.

![[Pasted image 20260418142007.png|400]]


In Geospatial, it's commonly used for [[InSAR|Interferometric Synthetic Aperture Radar]] (InSAR)
- We compare two [[Synthetic Aperture Radar|SAR]] radar images of the same area, taken at different times
	- Same area, two passes gives two phase measurements.
	- We can subtract the phases to get an ==interferogram== showing phase difference.
	- Phase difference encodes how much the ground moved between passes.
	- We can detect *millimeter-scale surface deformation* over hundreds of kilometers!

Used for:
- Earthquake deformation mapping
- Volcano inflation/deflation
- Subsidence (cities sinking, Venice/Mexico City/Jakarta)
- Glacier flow
- Landslide precursor detection
- Permafrost freeze/thaw cycles


This is what we used ot measure einstein's gravitational waves! It's so precise, that we can produce measuremnts down to 1/100000th of a proton! the smallest measurement attempted by sciencec, ever!
- Radar satellites can't do this, but hte same principle allows us to measure andm ap the deformation of the ground after earthquakes, or large explosions, or see track marks left b cars as they drive through fields, or measure hte thermal expansions of buildings from space as they expand/dialate from temperature difference -- all of that from fricken spaec!




