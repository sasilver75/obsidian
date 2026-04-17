---
aliases:
  - LiDAR
  - Laser Altimeter
---
An active [[Remote Sensing]] technology that ==measures distance by emitting laser pulses and timing how long they take to return==, which straightforwardly converts to distance.

The result is not an image, but a ==point cloud==; a collection of millions or billions of 3D points, each representing where a laser pulse reflected back from a surface.

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
	- [[ICESat-2]] (2018): 
	- [[Global Ecosystem Dynamics Investigation]]













