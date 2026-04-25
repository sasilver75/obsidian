---
aliases:
  - Shading
---
A shader is a small program that runs on the GPU, written in [[OpenGL Shading Language]] (GLSL), a C-like language designed for parallel numeric computation.
- When a GPU draws a 3D scene, they run two tiny programs for each frame: a [[Vertex Shader]] (runs once per corner of every triangle, deciding on screen where that corner ends up) and a [[Fragment Shader]] (runs once per pixel inside the triangle, deciding what color the pixel is). They're written in [[OpenGL Shading Language]] ([[OpenGL Shading Language|GLSL]]), a C-like language.
- [[Three.js]] ships with pre-built shaders like MeshStandardMaterial, MeshBasicMaterial, so you don't usually have to write GLSL yourself.

The name comes from their original purpose, computing how light *shades* a surface.
- Today, they're used for any per-vertex or per-pixel computation.

```glsl
// A minimal vertex shader
attribute vec2 position;   // input: one vertex's coordinates
uniform mat4 transform;    // input: the current zoom/pan matrix (same for all vertices)

void main() {
  gl_Position = transform * vec4(position, 0.0, 1.0);
  // gl_Position is the output: screen coordinates for this vertex
}
```

Critically, ==shaders run in parallel==; if you have 10,000 hexagons, the GPU runs 10,000 [[Vertex Shader]] instances simultaneously -- one per vertex.
- This is why [[WebGL]] is fast, whereas CPU would be slow.

[[Vertex Shader]]: Run once per vertex.
- Input: Coordinates + any per-feature data
- Output: Screen position

[[Fragment Shader]]: Runs once per pixel.
- Input: Interpolated values form nearby vertices
- Output: RGBA color.



![[Pasted image 20260425114000.png]]

![[Pasted image 20260425113742.png]]



