---
aliases:
  - SR
---
==The fraction of the incoming sunlight that's actually reflected== by the Earth's surface itself (grass, soil, water, buildings) ==after ***removing the effects of atmosphere.***==

When a sensor captures light, it sees sunlight that's been scattered, absorbed, and distorted by the atmosphere (aerosols, water vapor, ozone, etc.). 
- This is called [[Top-of-Atmosphere]] reflectance.
- ==Two identical fields of wheat photographed on different days with different atmospheric conditions will look different even though the surface hasn't changed.==

Why it matters for analysis:
- Change detection: Compare same area across different dates/sensors reliably
- Spectral indices: [[Normalized Difference Vegetation Index|NDVI]], [[Normalized Difference Water Index|NDWI]], [[Normalized Burn Ratio|NBR]] all assume you're working with [[Surface Reflectance|SR]], not [[Top-of-Atmosphere|TOA]]
- Multi-sensor fusion: Combine [[Landsat]] + [[Sentinel]] data meaningfully
- Machine learning: ==Models trained on [[Surface Reflectance|SR]] generalize across scenes==, while [[Top-of-Atmosphere|TOA]] models do not.

Common SR products:
- Landsat Collection 2 Level-2 (USGS)
- Sentinel-2 LSA (ESA, processed with Sen2Cor)
- Both available via [[SpatioTemporal Asset Catalog|STAC]] catalogs, accessible with rasterio/stackstac

The tradeoff:
- [[Surface Reflectance|SR]] requires accurate ==atmospheric data== at the time of the acquisition. ==If that's unavailable or low quality, TOA might actually be more reliable;  SR correction can introduce its own errors.==


Going from [[Top-of-Atmosphere|TOA]] to [[Surface Reflectance|SR]] requires ==atmospheric correction==, modeling what the atmosphere did to light and *inverting it*.
- Light from sun hits the atmosphere, some gets scattered/absorbed before reaching the surface, the surface reflects some fraction, then that reflected light travels back through the atmosphere again, getting scattered/absorbed a second time before reaching the sensor.

Sun -> atmosphere -> Surface -> atmosphere -> Sensor

The sensor sees a mix of:
1. Light reflected by the surface (what we want)
2. ==Path radiance==: Light scattered by the atmosphere directly into the sensor without ever hitting the surface
3. ==Adjacency effects==: Light reflected from neighboring pixels scattered into the sensor


What you need to model atmosphere:
- Aerosol Optical Depth (AOD): How much aerosol (dust, smoke, haze) is present 
- Water vapor: How much moisture in the atmosphere
- Ozone: Absorbs certain wavelengths
- Solar/view geometry: Sun angle, sensor angle at time of acquisition

Common approaches:
- Dark Object Subtraction (DOS)
	- Assumes the darkest pixel in the scene (deep water, deep shadow) has ~0 surface reflectance. Whatever signal sees there must be pure atmospheric scattering (path radiance). Subtract that from every pixel.
	- Fast but crude
- MODTRAN/6S/libRadtran
	- Physics-based simulation of how light travels through atmosphere; much more accurate but requires auxiliary atmosphere data.
- LaSRC ([[Landsat]])
	- Uses [[Moderate Resolution Imaging Spectroradiometer|MODIS]]-derived aerosol data, applies 6S-based radiative transfer
- Sen2Cor ([[Sentinel|Sentinel-2]])
	- Similar approach to above, uses scene-derived atmospheric estimates

Correction per band looks *something like:*
```
  SR = (TOA - path_radiance) / (atmospheric_transmittance_down ×
  atmospheric_transmittance_up)
```

