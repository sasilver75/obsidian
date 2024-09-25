September 25, 2024
[[Meta AI Research]]
Blogpost: [Llama 3.2: Revolutionizing edge AI and vision with open, customizable models](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/?utm_source=twitter&utm_medium=organic_social&utm_content=video&utm_campaign=llama32)


Building on the release of LLaMa 3.1, which included the interesting 405B model that challenged latest GPT-4 and Claude 3.5 versions, this adds some multimodal models to Meta's open-weights lineup.
- 11B and 90B vision language models (VLMs)
	- Competitive with Claude 3 Haiku/GPT4o-mini
- 1B and 3B text-only models for edge/mobile devices (both pretrained and SFT'd), with special attention to tool-calling abilities (More basic version of we've seen in the Apple Intelligence on-device models, but without the hot-swappable LoRAs). Result of large model distillation (from the 3.1 8B and 70B models) and pruning.
	- Competitive with Gemma 2 2.6B and Phi 3.5-mini
- Some discussion about the LLaMA Stack API, a "standardized interface for canonical toolchain components" around finetuning/generating synthetic data with LLaMA models.
- [[LLaMa Guard 3]] 11B Vision, which is a "guardrail model" designed to support LLaMA 3.2's image understanding capability
- LLaMA Guard 3 1B, for use with the text-only on-device models.