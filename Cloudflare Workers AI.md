[[Cloudflare]]'s serverless AI inference platform. It ==lets you run ML models directly from a [[Cloudflare Workers|Worker]] via a binding==, without managing GPUs or external API calls.

Key points:
- ==Runs on Cloudflare's GPU network==: Models execute on GPUs distributed across Cloudflare's ==edge== data centers, not on your Worker's CPU.
- Model Catalog: Pre-deployed open models: Llama, Stable Diffusion, [[Whisper]], embedding models, reranking, text classification, image-to-text, etc.
- Pricing: "Pay per 'neuron' (Cloudflare's compute unit), with a ==free daily allowance.==
- ==AI Gateway==: Separate but related product: a proxy in front of Workers AI *or* third party providers (OpenAI, Anthropic, etc.) that adds caching, rate limiting, analytics, fallbacks.
- ==Vectorize==: A companion vector database for [[Retrieval-Augmented Generation|RAG]], queryable from the same Worker.

Competes conceptually with [[Vercel AI Gateway]] or [[Amazon Bedrock|AWS Bedrock]]

