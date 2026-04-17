---
aliases:
  - MGRS
---
A geocoordinate standard used by NATO militaries to reference locations on the earth's surface. 
- It's built on the [[Universal Transverse Mercator]] (UTM) projection and the [[Universal Polar Stereographic]] (UPS) system for polar regions.
- [[Sentinel|Sentinel-2]] imagery is distributed using MGRS tile names (Each Sentinel-2 tile is named by its GZD + 100km square identifier: `t33UXP_20230615_B04.tif`)
	- The fact that Sentinel-2 adopted MGRS tiles is somewhat pragmatic - 100km squares in UTM are a convenient, fixed-size unit for distributing satellite imagery.
- It's not a coordinate system, NOR is it a [[Discrete Global Grid System|DGGS]], instead it provides a structured, human-readable way to express [[Universal Transverse Mercator|UTM]] coordinates as alphanumeric strings. It's more of an addressing convention (naming scheme) than a coordinate system (like UTM).

Rather than using decimal degrees (lat/lng), MGRS gives every point on Earth on Earth ==an alphanumeric address that encodes its location hierarchically==, from broad region down to meter-level precision.
```
33UXP 12345 67890
```
- This breaks down as `zone -> band -> square -> easting -> northing`

The hierarchy:
- ==Grid Zone Designator (GZD)==: The world is divided into 6 degree longitude by 8 degree latitude cells. Each cell gets a number (1-60) and a letter (C-X, excluding I and O). So `33U` means "longitude zone 33, latitude band U."
	- This gives a cell roughly 668,000km^2, a large region (e.g. Central Europe)
- ==100km Square Identifier==: Within each GZD, a pair of letters identifies a 100km x 100km square. These letters cycle in a defined pattern across the grid. `XP` in `33UXP` identifies one specific 100km square within zone `33U`.
- Numerical Coordinates: Within the 100km square, a numerical easting and northing give position. The precision depends on how many digits you use:
```
33UXP 1 6         → 10km precision
33UXP 12 67       → 1km precision
33UXP 123 678     → 100m precision
33UXP 1234 6789   → 10m precision
33UXP 12345 67890 → 1m precision
```






















