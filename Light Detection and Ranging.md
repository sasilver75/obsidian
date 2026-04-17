---
aliases:
  - LiDAR
  - Laser Altimeter
---
An active [[Remote Sensing]] technology that ==measures distance by emitting laser pulses and timing how long they take to return==, which straightforwardly converts to distance.

The result is not an image, but a ==point cloud==; a collection of millions or billions of 3D points, each representing where a laser pulse reflected back from a surface.
- Data formats
	- [[LASer|LAS]]: Standard binary format for LiDAR point clouds, maintained by [[American Society for Photogrammetry and Remote Sensing|ASPRS]]
	- [[LAZ]]: Compressed [[LASer|LAS]]; the same structure but with lossless compression applied, with typically compression ratios of 5-10x over LAS. LAZ has become the de facto standard for storage and transfer: there's no reason to store LAS anymore, and modern tools read LAZ natively.
	- Both of these file formats are not cloud-native; same problem as unoptimized [[GeoTIFF]]s, for instance: Reading a spatial subset requires either downloading the whole file or having an external spatial index, resulting in creation of:
		- [[Cloud-Optimized Point Cloud]] (COPC): A LAZ file reorganized as a spatial octree (3d equivalent of a [[QuadTree]]), with chunk locations stored in a VLR header. HTTP range requests can fetch just the points in a spatial region. Same philosophy and single-file approach as [[Cloud-Optimized GeoTIFF|COG]].
		- [[Entwine Point Tiles]] (EPT): A Potree-compatible tiling format that stores point clouds as a hierarchical octree of LAZ files in a directory structure. Similar philosophy as [[Zarr]]: many files rather than one. Used by [[United States Geological Survey|USGS]]'s [[3DEP]] for distributing national LiDAR data.

A LiDAR system fires laser pulses at rates from ==tens of thousands to millions of pulses a second==; as the sensor moves (e.g. on an aircraft, drone, or satellite), it sweeps the laser across the terrain using a rotating mirror or oscillating prism, ==building up dense 3D coverage==.

Each pulse returns:
- ==X, Y position== of the sensor at moment of firing (derived from [[Global Positioning System|GPS]] + [[Inertial Measurement Unit|IMU]] orientation of sensor)
	- The GPS + IMU combination is called "==direct georeferencing==" and is what converts raw range measurements into real-world coordinates; knowing exactly where the sensor is is critical to the accuracy of the final point cloud.
- ==Z elevation==, derived from the time of flight
- ==Intensity==: How strongly the pulse reflected, which encodes surface properties
- ==Return number==: Crucially, one pulse can generate *multiple returns*.
	- When a laser hits a tree canopy, some energy reflects from top leaves (***first return***), some penetrates through gaps and reflects from branches in the middle (***intermediate returns***), and some reaches the ground (***last return***).
	- A single pulse can generate 2-5 or more discrete returns, each recorded separately.

This ==ability to penetrate and create multiple returns== makes LiDAR transformative for forestry and terrain mapping; passive optical imagery sees only the top of the canopy. 
- No other remote sensing technology does this reliably at scale.

The last return in vegetated areas is typically used for bare-earth [[Digital Terrain Model]] generation, while the first return can be used for [[Digital Surface Model]] generation. The difference in the two can give you the [[Canopy Height Model]] (CHM).

# Full Waveform vs Discrete Return
- ==Discrete return== systems record distinct peaks in the return signal; this is the ==standard approach==, and produces individually labeled points.
- ==Full waveform== systems digitize the entire return signal as a continuous waveform; it's much more information/data volume, but requires more complex processing. It's common in research and forestry applications, because it encodes information about vegetation structure.


Platforms:
- ==Airborne LiDAR==: Dominant platform; sensors on fixed-wing aircraft or helicopters, typically flying at 500-3000m altitude, with point densities of 1-50+ points/m^2 typical, and swath widths of 100-2000m.
	- Most national LiDAR programs like [[United States Geological Survey|USGS]]'s [[3DEP]] use airborne platforms.
- ==Drone LiDAR== (UAS): Smaller sensors on UAVs, flying low (30-150m). Produces very high density point clouds (100-1000+ pts/m^2). Limited area coverage per flight, but exceptional detail. Used for precision agriculture, infrastructure inspection, archaeological survey, construction monitoring.
- Terrestrial LiDAR (TLS): Ground-based tripod-mounted scanners, used for detailed survey of buildings, infrastructure, forests, archaeological sites, quarries. Can be extremely high density, but generally limited spatial extent.
- Mobile LiDAR (MLS): Sensors mounted on vehicles, used for road corridor mapping, utility mapping, highway asset inventory. Google Street View imagery is accompanied by mobile LiDAR data.
- Satellite LiDAR (two notable missions):
	- ICESat-2 (2018): Uses photon-counting LiDAR, firing 10,00 pulses at 532nm green wavelength. Designed for ice sheet evaluation change but used for vegetation height globally. Not imaging LiDAR; produces along-track profiles, not area coverage.
	- Global Ecosystem Dynamics Investigation (2018): Full waveform LiDAR specifically designed for global forest structure and biomass mapping. Samples are not contiguous; footprints are spaced 60m along tracks, and tracks spaced 600m apart globally.



# Point Clouds
- ==Point density matters enormously==  for what you can extract; at 1pt/m^2 you can make a decent [[Digital Elevation Model|DEM]]; at 8 pts/m^2 you can map a building reliably; at 25+ pts/m^2 you can extract individual trees and measure their dimensions.
- Point Cloud Characteristics
	- XYZ coordinates (usually in a projected [[Coordinate Reference System|CRS]], meters)
	- Intensity (0-65535 typically; 16-bit)
	- Return number and number of returns per pulse
	- Scan angle
	- GPS time of acquisition
	- RGB color (if sensor is paired with a camera; called a "colorized point cloud")
	- ==Classification== (discussed below)
- Classification: Raw LiDAR points need to be classified; the [[American Society for Photogrammetry and Remote Sensing]] (ASPRS) defines standard classification codes:
```
Class 0:  Never classified
Class 1:  Unassigned
Class 2:  Ground
Class 3:  Low vegetation (0-0.5m)
Class 4:  Medium vegetation (0.5-2m)
Class 5:  High vegetation (>2m)
Class 6:  Building
Class 7:  Low point (noise)
Class 9:  Water
Class 17: Bridge deck
Class 18: High noise
```
- Tools for classification include: LAStools (proprietary, vrey fast, industry standard), PDAL (open source pipeline library), lidR (R package for forestry applications), whitebox-tools (open source), CloudCompare.
















