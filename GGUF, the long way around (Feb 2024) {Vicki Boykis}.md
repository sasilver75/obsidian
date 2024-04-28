#article 
Link: https://vickiboykis.com/2024/02/28/gguf-the-long-way-around/

---

How we use LLM artifacts:
1. As an API endpoint for proprietary models hosted by OpenAI/Anthropic/major cloud providers
2. As model artifacts downloaded from HuggingFace's model hub and/or trained/fine-tuned using HuggingFace libraries, and hosted on local storage.
3. As model artifacts available in some format optimized for local inference, typically ==GGUF==, and accessed via applications like `llama.cpp` or `ollama`
4. As [[ONNX]], a format that optimizes sharing between backend frameworks.