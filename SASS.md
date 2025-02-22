---
aliases:
  - Streaming Assembler
---
- SASS is the assembly format for programs running on NVIDIA GPUs.
- ==This is the lowest-level format in which human-readable code can be written.==
	- It's one of the formats output by [[nvcc]], alongside [[PTX]].
- It is converted to device-specific binary microcodes during execution.
- Presumably, the "Streaming" in "Streaming Assembler" refers to the [[Streaming Multiprocessor]]s which the assembly language programs.
- SASS is versioned and tied to a specific NVIDIA GPU SM architecture.