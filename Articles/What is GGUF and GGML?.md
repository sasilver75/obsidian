#article 
Link: https://medium.com/@phillipgimmi/what-is-gguf-and-ggml-e364834d241c

Review: This isn't a very good article.

----

[[GGUF]] and [[GGML]] are file formats used for storing models for inference, especially in the context of language mdoels like GPT.

-----

GGUF is a File Format (also a tensor library designed for ML, facilitating large models and high performance on various hardware, including Apple Silicon)

Pros:
- Early innovation: An early attempt to create a file format for GPT models
- Single file sharing: Models are a single file, enhancing convenience
- CPU Compatability: Can run on CPUs, broadening accessibility

Cons:
- Limited Flexibility: GGML struggled with adding *extra information* about the model
- Compatibility issues: Introduction of new features often led to compatability problems with older models
- Manual Adjustments Required
	- Users frequently had to modify settings like rope-freq-base, rope-freq-scale, gqa, rms-norm-eps, which can be conmplex.

---

GGML: A file format for models introduced as a successor to GGML, released in August 2023.

Represents a step forward, facilitating enhanced storage and processing of large language models like GPT.

Developed b y contributed from the AI community including Georgi Gerganov, the creator of [[GGML]].
Its use in contexts involving Facebook's [[LLaMA]] models underscores its importance in the AI landscape.


Pros:
- Addresses GGML limitations, enhances user experience
- Extensibility: Allows for the addition of new features while maintaining compatability with older models
- Stability: Focuses on eliminating breaking changes, easing the transition to newer models.
- Versatility: Supports various models, extending beyond the scope of llama models.

Cons:
- Transition time (Converting existing models to GGUF requires significant time)
- Adaptation required (Users and developers must become accustomed to the new format)


