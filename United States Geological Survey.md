---
aliases:
  - USGS
  - USGS National Map
---

Maintains ==USGS National Map== ([nationalmap.gov](https://nationalmap.gov), USGS'S primary platform for distributing authoritative US geospatial data.
- Contains:
	- [[3DEP]] elevation data
	- National Hydrography Dataset (NHD): Authoritative US water network
	- Watershed Boundary Dataset (WBD): Hydrologic unit boundaries at multiple scales
	- National Land Cover Database (NLCD): 30m land cover classification updated every few years
	- National Structures Dataset: Buildings, landmarks
	- Transportation: Roads, trails, railways
	- Geographic Names: Official place names (GNIS)
	- Orthoimagery: Aerial photography
- Accessible by Web map viewer at nationalmap.gov, but also via the TNM Download API (programmatic access) and The National Map Downloader (build download application).




Hosts [[USGS EarthExplorer]], a website... 
![[Pasted image 20260422122606.png]]
![[Pasted image 20260422123842.png]]
Can specify a ==location, time==, etc. For Greenland, maybe we want to look at Greenland in the summer, so we look from April 1 2024 to the end of summer, September 30, 2024.
-  Can also specify what level of cloud cover you're willing to accept.
	- 0% is relatively uncommon, so say: 0-20% is a good option.
	- You can filter how many options you want per returned page (e.g. 10)
![[Pasted image 20260422124011.png]]
In the next tab, you can specify the image sets that you want to look through/for.
![[Pasted image 20260422124100.png]]
- Here, we might wantto look at Landsat
- There are many collection levels to look from, so we'll look Level-2 ([[Surface Reflectance]]), and select Landsat-8.
![[Pasted image 20260422124106.png]]
There are even additional criteria for what we might want to download
- If you know the name of the fine
- If you know the WRS path/row (see [[Landsat]])
- Can specify whether Landsat-8 or Landsat-9
- Can specify if you only care about the optical or thermal images

![[Pasted image 20260422124306.png]]
Finally, we can get the results, and click on them:
![[Pasted image 20260422124317.png]]
- See that there's some cloud cover in top-right corner, but rest of image is pretty clear.

