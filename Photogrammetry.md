---
aliases:
  - Stereophotogrammetry
  - Photogrammetric
  - Multi-View Stereo
  - Structure from Motion
  - SfM
  - MVS
  - Oblique Photogrammetry
---
==The science of extracting geometric measurements and 3D information from photographs.==
- If you photograph the same object or scene from multiple positions, you can use the geometry of overlapping images to calculate the 3D position of any point that appears in more than one image. This is essentially how your brain uses two eyes (stereo) to perceive depth, but done mathematically with cameras!

You can produce:
- Dense [[Point Cloud]]s
- [[Digital Elevation Model|DEM]]s and [[Digital Surface Model|DSM]]s
- [[Orthoimage|Orthophotomosaic]]s
- 3D meshes

How it works at a high level:
1. Identifying matching features across overlapping images
2. Using those correspondences to solve for camera positions and orientations ([[Photogrammetry|Structure from Motion]]/SFM)
3. Project rays from each camera through matched points; where rays intersect is the 3D location (stereo reconstruction)

Where it shows up:
- Satellite stereo imagery (two passes over the same area from different angles, or two offset cameras on a boom, sort of like in [[Shuttle Radar Topography Mission|SRTM]], leading to a [[Digital Elevation Model|DEM]])
- Aerial surveys: Overlapping aircraft imagery leading to an [[Orthoimage|Orthomosaic]] and [[Digital Elevation Model|DEM]]
- Drone mapping: The dominant use case today, very accessible
- DigitalGlobe/Maxar's stereo products are photogrammetric DEMs.


==Stereophotogrammetry==: Involves estimating the three-dimensional coordinates of points on an object employing measurements made in two or more photographic images, taken from different positions. ==The classical approach.==

==[[Photogrammetry|Structure from Motion]]] (SfM)==: ==The modern approach==, many unordered images from unknown positions, simultaneously solves for camera poses and 3D structure. What drone mapping software uses.

==[[Photogrammetry|Multi-View Stereo]]== (MVS): Usually follows SfM, densifies the sparse point cloud into a dense representation.

==[[Photogrammetry|Oblique Photogrammetry]]==: Cameras pointed at angles (not just straight down), captures building facades for true 3D city models, used in products like Nearmap and Vexcel.

The ==modern pipeline== in practice is almost always ==[[Photogrammetry|SfM]] -> [[Photogrammetry|MVS]] -> Mesh/DEM/Orthomosaics==, and the line between photogrammetry and computer vision has largely dissolved; they're the same math now. For this reason, sometimes they're written together as ==SfM-MVS==.


![[Pasted image 20260424220437.png]]
Above: GPT-Image 2: "Explain to me how modern photogrammetry works"
- Take many photos from varying, overlapping angles.
- Software detects keypoints in each photo, finds overlapping points across multiple images.
- Estimate where the camera poses are, and create a sparse point cloud of the scene.
- Now compare all images to compute depths for millions of pixels. 
	- By finding the same points in multiple images, the software can triangulate their 3D positions, just like our eyes do.


![[Pasted image 20260424220915.png]]
Above: GPT-2 image, explain those steps in more detail
