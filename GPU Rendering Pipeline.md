When you render geometry on the GPU (say, using [[WebGL]] in [[deck.gl]] or [[MapLibre GL JS]]), data flows through a pipeline:

```
Your data (coordinates, colors)
        ↓
Vertex Buffer — flat array of numbers uploaded to GPU memory
        ↓
Vertex Shader — runs once per vertex, transforms coordinates
  (e.g. lat/lon → screen pixels)
        ↓
Rasterization — GPU figures out which pixels each triangle covers
        ↓
Fragment Shader — runs once per pixel, outputs a color
  (e.g. color based on count value)
        ↓
Framebuffer — the final image
        ↓
Canvas → Screen
```
Above:
- [[Vertex Shader]]: A tiny program that takes one vertex (a point in space) and outputs its screen position.
	- For a map, this means converting geographic coordinates (lon/lat) into screen pixels, based on the current zoom/pan.
- [[Fragment Shader]]: A tiny program that takes a pixel position and outputs an RGBA color. This is where you implement effects like color ramps (count=100 -> orange; count=500 -> red).

