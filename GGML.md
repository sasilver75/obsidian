---
aliases:
  - GPT-Generated Model Language
---
A File Format (also a tensor library designed for ML, facilitating large models and high performance on various hardware, including Apple Silicon)

c.f. [[GGUF]], the successor to GGML

Pros:
- Early innovation: An early attempt to create a file format for GPT models
- Single file sharing: Models are a single file, enhancing convenience
- CPU Compatability: Can run on CPUs, broadening accessibility

Cons:
- Limited Flexibility: GGML struggled with adding *extra information* about the model
- Compatibility issues: Introduction of new features often led to compatability problems with older models
- Manual Adjustments Required
	- Users frequently had to modify settings like rope-freq-base, rope-freq-scale, gqa, rms-norm-eps, which can be conmplex.