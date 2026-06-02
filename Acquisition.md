The single instance of a sensor collecting data over an [[Area of Interest]]: one imaging event.
- When you query a data catalog ([[SpatioTemporal Asset Catalog|STAC]], [[Google Earth Engine]], [[Microsoft Planetary Computer]]), you're searching for acquisitions that match your criteria; a scene at a moment.

When a satellite "acquires" an area, that encompasses all of:
1. The sensor is [[Task|Tasked]] to collect at a specific time/location
2. The satellite may maneuver/[[Slew]] (changing [[Attitude]]) to point at the target.
3. Data is collected during the [[Overpass Window]].
4. Raw data is [[Downlink]]ed to a [[Ground Station]]
5. Data is processed into an image product

Defined by:
- Datetime: When the sensor collected
- Geometry: View angle, sun angle, [[Nadir|Off-Nadir]] angle
- Cloud cover %: Usability for optical
- Resolution: Which imaging mode was used
- Polarization: For [[Synthetic Aperture Radar|SAR]], whether [[Synthetic Aperture Radar|HH]], [[Synthetic Aperture Radar|VV]], [[Synthetic Aperture Radar|HV]], etc.
- Incidence angle: For SAR, this impacts the [[Backscatter]] interpretation.

## Satellite Operational Modes: Tasked vs Systematic Acquisition
1. ==Systematic==: Satellite collects everything in its swath continuously, regardless of specific requests. [[Sentinel|Sentinel-1]] and [[Sentinel|Sentinel-2]] operate this way, they just image everything on a fixed schedule.
2. ==Tasked==: A customer or operator directs the satellite to a specific target. This is how [[Maxar]]'s [[Maxar|WorldView]] operates: someone pays to point the satellite at a location (finite resource, competing requests, priority queueing).