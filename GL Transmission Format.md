---
aliases:
  - glTF
---
An open 3D file format form the Khronos Group designed for efficient runtime delivery of 3D content to the GPU. Often called ==the [[Joint Photographic Experts Group|JPEG]] of 3D==.

It's a [[JSON]] file describing the scene (meshes, materials, cameras, animation) and a `.bin` binary file with raw geometry data and texture images.
- GLB is the single-file binary variant that packs all three into one file.
- Uses [[Physically-Based Rendering]] (PBR) for physically realistic rendering

# Relevance
- [[3D Tiles]] 1.1 uses [[GL Transmission Format|glTF]] as its universal tile content format.
- Some [[Photogrammetry]] outputs (from Metashape, RealityCapture) export as glTF
- [[Building Information Modeling|BIM]] -> [[Industry Foundation Classes|IFC]] -> [[GL Transmission Format|glTF]] is a common conversion pipeline for web visualization.
- [[Cesium]], [[Three.js]], and [[deck.gl]] all load glTF natively.


