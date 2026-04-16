---
aliases:
  - TIR
---
Bandwidth: ~8,000 - 14,000 nm (8-14 um)

A completely different physical phenomenon from [[Near Infrared|NIR]] and [[Short-Wave Infrared|SWIR]], which measure reflected sunlight. Instead, [[Thermal Infrared|TIR]] ==measures emitted thermal radiation==, heat that objects radiate based on their own temperature.

==Every object above absolute zero emits in this range, hotter objects just emit more.==

![[Pasted image 20260415203626.png|400]]

This is how you get [[Land Surface Temperature]] (LST) estimates from satellites like [[Landsat]].
- Asphalt, metal roofs, and concrete absorb heat and re-emit it strongly.
- Vegetated surfaces stay cooler through evapotranspiration.
- Water has a high thermal mass and changes temperature slowly.

TIR sensors require cooling to detect the faint thermal emissions from the ground above their own sensor noise, making them ==heavier==, ==more expensive==, and ==lower resolution than optical sensors==. Landsat's thermal band is 100m vs 30m for its optical bands.

[[Sentinel|Sentinel-2]] has *no thermal band.*
You need [[Landsat]], ASTER, or ECOSTRESS for surface temperature work.



