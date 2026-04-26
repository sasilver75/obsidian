---
aliases:
  - PBR
---

PBR is a [[Shader|Shading]] model that ==approximates how light actually interacts with surfaces using physically meaningful parameters.==

The goal is that a ==material defined once looks correct under any lighting condition== - 
sunglight, indoor lighting, overcast sky, etc. because it's ==described in terms of *real physical properties rather than arbitrary artistic tweaks!==*

# Geospatial Relevance
- For [[Photogrammetry]] meshes and [[Building Information Modeling|BIM]] materials, you have textures that need to be rendered plausibly under a variety of lighting conditions.
	- BIM materials have known physical properties that map naturally to PBR parameters.
- When you bring a building model into [[Cesium]] and the sun moves across the sky, PBR materials respond correctly to the changing illumination direction.


![[Pasted image 20260425201750.png]]