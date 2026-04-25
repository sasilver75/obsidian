(Usually called Spark, just named Spark JS here because I don't want to shadow with [[Apache Spark]].)

An advanced [[3D Gaussian Splatting]] renderer for [[Three.js]], built by World Labs.

[Examples](https://sparkjs.dev/examples/#hello-world)

In April 2025, released "Spark 2.0"
- The main driver for Spark 2.0 is to enable huge worlds made of dynamic 3D Gaussian Splats.
- The new Spark 2.0 is a complete solution for ==creating, streaming, and rendering huge [[3D Gaussian Splatting|3DGS]] worlds on the web on any device.== 
	- Any splat file can be loaded and turned into a =="***Level-of-Detail (LoD) Splat Tree==***" with all of the original splats as leaf nodes, and interior nodes representing downsampled versions of the splats all the way up to the top "root splat" that has the average color and shape of all the original splats combined.
	- As you move around the scene, Spark computes "slices" through this tree that picks the best set of N splats from your current viewpoint, taking into account your distance to each splat and the view frustum.
	- Supports LoD rendering across *multiple splat objects* simultaneously, traversing multiple trees jointly to compute the optimal set of splats that maximizes the maximum screen-space splat sizes. This lets us create huge composite worlds by adding as many splat object parts as you want to.
	- Ships with two selectable algorithms for computing LoD splat trees, a quick and compact algorithm (`tiny-lod`) (intended to be run on demand) and a higher-quality `bhatt-lod` (intended for pre-processing for faster load times). Both can be run using a command-line tool or directly in the browser.
	- Streamable LoD file format: Spark also defines a new, extensible, and configurable file format (`.RAD`; RADiance file) that can store the precomputed LoD splat tree, and enables streaming arbitrary chunks of splats via HTTP range requests.
	- To manage huge memory usage of composite worlds, Spark 2.0 implements a shared LRU "splat page table" (like an OS/CPU virtual paging system) that pre-allocates a fixed GPU memory pool for splats that's shared across all splat objects, fetching splats over the network + evicting the oldest or least useful chunks for the current viewpoint.
		- Scenes with 100M or 1G+ splats can be rendered in real-time on mobile devices with limited GPU memory, and Spark automatically prioritizes and manages it according to the current scene splat objects and viewpoints.





