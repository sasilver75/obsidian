October 2, 2023 (5 months after [[UltraChat]])
Tsinghua University
[UltraFeedback: Boosting Language Models with High-quality Feedback](https://arxiv.org/abs/2310.01377)

Takeaway: ...

Note: When training [Notus 7B](https://huggingface.co/argilla/notus-7b-v1) (an "improvement" vs. [[Zephyr]]-Beta), the authors from [[Argilla]] noticed some issues in the original UltraFeedback dataset, leading to high-scores for bad responses... they manually-curated several hundreds of data points, and then binarized the dataset (for [[Direct Preference Optimization|DPO]]) and verified it with the Argilla platform... It led to a new dataset where the chosen response is different in ~50% of cases! This dataset is named [ultrafeedback-binarized-preferences](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences) and is available on the hub.

---


A ==synthetic dataset== of "human preferences" (really well-prompted [[GPT-4]] preference judging, a la [[LLM-as-a-Judge]]; GPT-4 is shown to be very correlated with human preferences, I believe).
- Generate 4 different responses from 4 language models; GPT-4 picks the winner.

They train an "UltraRM" chat language model: UltraLM-13B-PPO, and a critique model ULTRACM.

Abstract
> Reinforcement learning from human feedback (RLHF) has become a pivot technique in aligning large language models (LLMs) with human preferences. ==In RLHF practice, preference data plays a crucial role== in bridging human proclivity and LLMs. However, the ==scarcity of diverse, naturalistic datasets of human preferences== on LLM outputs at scale poses a great challenge to RLHF as well as feedback learning research within the open-source community. ==Current preference datasets, either proprietary or limited in size and prompt variety,== result in limited RLHF adoption in open-source models and hinder further exploration. In this study, we propose ==ULTRAFEEDBACK==, a ==large-scale, high-quality, and diversified preference dataset designed to overcome these limitations and foster RLHF development==. To create ULTRAFEEDBACK, we compile a diverse array of instructions and models from multiple sources to produce comparative data. We meticulously devise annotation instructions and ==employ GPT-4 to offer detailed feedback in both numerical and textual forms==. ULTRAFEEDBACK establishes a reproducible and expandable preference data construction pipeline, serving as a solid foundation for future RLHF and feedback learning research. Utilizing ULTRAFEEDBACK, we train various models to demonstrate its effectiveness, including the reward model UltraRM, chat language model UltraLM-13B-PPO, and critique model UltraCM. Experimental results indicate that our models outperform existing open-source models, achieving top performance across multiple benchmarks. Our data and models are available at [this https URL](https://github.com/thunlp/UltraFeedback).

![[Pasted image 20240424153738.png]]




# Non-Paper Figures

![[Pasted image 20240717105802.png|600]]
An example of one poorly-provided human feedback in the UltraChat dataset noticed by Argilla when training Notus.