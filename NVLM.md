September 17, 2024
[[NVIDIA]]
Blog Post: [NVLM: Open Frontier-Class Multimodal LLMs](https://research.nvidia.com/labs/adlr/NVLM-1/)
Huggingface: [Link](https://huggingface.co/nvidia/NVLM-D-72B)
Arxiv: [NVLM: Open-Fronteir Class Multimodal LLMs](https://arxiv.org/abs/2409.11402)

Model sizes:
- 72B


>> We introduce NVLM 1.0, a family of frontier-class multimodal large language models (LLMs) that achieve state-of-the-art results on vision-language tasks, rivaling the leading proprietary models (e.g., [[GPT-4o]]) and open-access models (e.g., ==Llama 3-V 405B== and ==InternVL 2==). ==Remarkably, NVLM 1.0 shows improved text-only performance over its LLM backbone after multimodal training==. In terms of model design, we perform a comprehensive comparison between decoder-only multimodal LLMs (e.g., LLaVA) and cross-attention-based models (e.g., Flamingo). Based on the strengths and weaknesses of both approaches, we propose a novel architecture that enhances both training efficiency and multimodal reasoning capabilities. Furthermore, we introduce a 1-D tile-tagging design for tile-based dynamic high-resolution images, which significantly boosts performance on multimodal reasoning and OCR-related tasks. Regarding training data, we meticulously curate and provide detailed information on our multimodal pretraining and supervised fine-tuning datasets. Our findings indicate that dataset quality and task diversity are more important than scale, even during the pretraining phase, across all architectures. Notably, we develop production-grade multimodality for the NVLM-1.0 models, enabling them to excel in vision-language tasks while maintaining and even improving text-only performance compared to their LLM backbones. To achieve this, we craft and integrate a high-quality text-only dataset into multimodal training, alongside a substantial amount of multimodal math and reasoning data, leading to enhanced math and coding capabilities across modalities. To advance research in the field, we are releasing the model weights and will open-source the code for the community: [this https URL](https://nvlm-project.github.io/).





# Blog Figures
![[Pasted image 20241006112716.png|500]]


![[Pasted image 20241006112728.png|450]]

![[Pasted image 20241006112825.png|400]]
