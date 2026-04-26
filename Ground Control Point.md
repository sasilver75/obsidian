---
aliases:
  - GCP
  - Control Point
---
A physical location on the Earth's surface whose precise real-world coordinates are known, and is used to [[Georeference]] (images -> coordinate) and [[Orthoimage|Orthorectify]] images (gemoetrically correct to remove tilt) and improve the absolute accuracy of spatial data.

Core problem:
- Satellites capture images. The image ahs pixels and those pixels need to be tied to real-world coordinates.You can use the sensor's GPS position and attitude (roll, pitch, yaw) to estimate where each pixel falls on the ground (this is called ==direct georeferencing==), though this has errors:
	- GPS Position errors
	- Attitude/orientation error from the [[Inertial Measurement Unit|IMU]] 
	- Terrain distortion - the sensor assumes flat Earth, but terrain has relief
	- Lens distortion
	- Atmospheric refraction
These ==errors accumulate== into positional inaccuracy, where a pixel that we labeled as being at (X,Y) might actually be at (X+5, Y+3). For many applications, that's unacceptable.

GCPs let us correct this.

## ==How GCPs work==
- Identify specific points that are:
	1. Clearly visible in the imagery
	2. Have precisely known real-world coordinates (surveyed with RTK GPS, dGPS, or total station; typically centimeter-level accuracy)
- Then tell your processing software:
	- *This specific pixel* in the image corresponds to this real-world coordinate.
	- The software uses these correspondences to compute a transformation that warps the image to match reality.

==Common GCP targets== in the field:
- Painted crosses, or checkboard targets placed before the flight
- Road lane markings, painted parking lot lines
- Building corners (if coordinates can be precisely measured)
- Survey markets


# Contexts

- GCPs are especially critical in ==Drone photogrammetry==, since Drone GPS is typically consumer-grade (1-3m accuracy). Without GCPs, a photogrammetric model might be geometrically internally inconsistent.
	-  Typically you have 5-10 GCPs distributed across the survey area
- ==Satellite Imagery== comes with ***[[Rational Polynomial Coefficient]]s (RPCs)***, a mathematical model describing the sensor geometry; the imagery is already *approximately georeferenced* from satellite GPS + start trackers (3-10m absolute accuracy). ***GCPs are used to refine the RPC model***, in a process called **==RPC refinement==*** or **==block adjustment==***.
- Aerial Survey: Traditional airborne photogrammetry has used GCPs. Modern aircraft with PPK (Post-Processed Kinematics) or RTK GPS can achieve direct georeferencing accuracy of 5-10cm without GCPs, but they're still used for quality control.
	- [[Real-Time Kinematic GPS]] (RTK GPS): A GPS technique using a base station at a known location broadcasting corrections to a rover receiver in real time, achieving 1-2cm horizontal accuracy. Used to survey GPS locations in the field.
	- [[Post-Processed Kinematic]] (PPK): Similar to RTK but corrections are applied after the fact in post-processing, rather than in real time. More flexible (no need for radio link between base and rover) and often more accurate.
	- A [[Total Station]] is a traditional surveying instrument that measures angles and distances from a known point with sub-centimeter accuracy. Used where GPS doesn't work well (under tree canopy, urban canyons).

GCPs at the edge and corners of your survey area are more valuable than GCPs clustered in the center. The transformation that maps image space to ground space needs to be constrained across the whole area; a cluster of GCPs in one corner leaves the opposite corner poorly controlled, potentially with large errors even if the reported accuracy looks good.

## Check Points vs Control Points:
- [[Ground Control Point|Control Point]]s are used in the processing/adjustment. The algorithm actively uses these to correct/warp the data.
- [[Independent Check Point]] (ICPs) are surveyed to the same accuracy as GCPs but deliberately withheld from processing. After processing, you compare the processed position of each check point to its surveyed position. This gives an independent, unbiased estimate of absolute accuracy. If your GCPs show 3cm error but your check points show 15cm error, something is wrong.
	- ((Sort of just a held-out control point that isn't used to actually adjust the image, just as a sanity check))


