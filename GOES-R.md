---
aliases:
  - GOES-16
  - GOES-17
  - GOES-18
  - GOES-19
  - Geostational Operational Environmental Satellite
---
Pronounced like "Ghost" without the T, I think.

The current generation of [[National Oceanic and Atmospheric Administration]]'s (NOAA) [[Geostationary Orbit|Geostationary]] weather satellites, representing a major upgrade over the previous GOES-NOP series.

Data is ==free and publicly accessible== on AWS [[Amazon S3|S3]] in NetCDF4 format, with [[Amazon SNS|SNS]] notifications on every new file drop, making it ==one of the best real-time satellite data pipelines available== to developers ==without authentication or cost==.

Four satellites (all active) and launch dates:
- ==GOES-16== (2016): Operational as GOES-East (75.2 W), covers eastern US, Atlantic, South America
- GOES-17 (2018): Operated as GOES-West but had a cooling system defect that degraded its ABI (advanced baseline imager, its primary imaging instrument) at night.
- ==GOES-18== (2022): Replaced GOES-17 as GOES-WEST (137.2 W), covers western US, Pacific, Alaska
- GOES-19 (2024): Spare/future replacement


## ABI: Advanced Baseline Imager
- Primary imaging instrument
- ==16 spectral bands covering visible through [[Thermal Infrared]] (TIR), at 500m-2km resolution depending on band==.
- Cadence:
	- Full disk (Western Hemisphere) every 10 minutes
	- CONUS: Every 5 minutes
	- Menoscale (targeted region): Every 30-60 seconds during events like fires or hurricanes.

This is the instrument you'd use if you were doing a smoke plume tracker project.

