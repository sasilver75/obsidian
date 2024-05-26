From [[Eleuther|EleutherAI]]
Github: [Link](https://github.com/EleutherAI/lm-evaluation-harness)

Project providing a unified framework to test generative language models on a large number of 200 different evaluation tasks.
- Over 60 standard academic benchmarks for LLMs (including [[BIG-Bench]], [[Massive Multi-task Language Understanding|MMLU]], and more), with hundreds of subtasks and variants implemented.
- Support for models loaded by `transformers`, with a flexible tokenization-agnostic interface, as well as commercial APIs like OpenAI
- Easy support for custom prompts and evaluation metrics

This is also the backend for [[HuggingFace]]'s popular [[OpenLLM Leaderboard]], and has been used in hundreds of papers, as well as internally at dozens of organizations.
