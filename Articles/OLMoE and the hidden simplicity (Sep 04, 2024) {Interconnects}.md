Link: https://www.interconnects.ai/p/olmoe-and-building-better-llms

---

[[Allen Institute|AI2]] just released [[OLMoE]], an MoE model with 1.3B active parameters, and 6.9B total parameters.
- Trained on 5T tokens, largely composed of the [[DataComp-LM|DCLM]] baseline mix.
- Apache 2.0 license for all model variants

![[Pasted image 20241008165118.png|500]]
![[Pasted image 20241008165123.png|400]]
![[Pasted image 20241008165154.png|500]]

Small models beginning to respond well to fine-tuning

---

On frontier model organizations

> "I’ve heard from many friends who have gone to top language model lives that their work can be seen as somewhat routine, with simple tasks, yet with incredible pace. ==There are a lot of people who are just grinding on making datasets all the time and a smaller percentage who are launching the biggest training runs==."

>All of the top labs have undergone substantial ==internal politics over compute allocations==. The biggest allocation goes to the fewest people — pretraining researchers. This ends up being about 60%, or some number greater than half. ==The full distribution is something like pretraining — 60%, post-training — 25%, data — 10%, and other — 5%.==
- Post-training percentage here is very high relative to the original landscape of ChatGPT.
	- When doing something like [[Rejection Sampling]] or another online method, it's normal to generate 10 to 30 or more samples per prompt.
		- Using a bigger model, like a [[LLaMA 3]] 405B model to do this, so that you can "distill" into your candidate model, will be more expensive. Filtering takes time too.

> Meta stated clearly in their [Llama 3 paper](https://arxiv.org/abs/2407.21783) that efficiency and simplicity were their core principles
> "> We make design choices that seek to maximize our ability to scale the model development process. For example, we opt for a standard dense Transformer model architecture (Vaswani et al., 2017) with minor adaptations, rather than for a mixture-of-experts model (Shazeer et al., 2017) to maximize training stability. Similarly, we adopt a relatively simple post-training procedure based on supervised finetuning (SFT), rejection sampling (RS), and direct preference optimization (DPO; Rafailov et al. (2023)) as opposed to more complex reinforcement learning algorithms (Ouyang et al., 2022; Schulman et al., 2017) that tend to be less stable and harder to scale."
> ![[Pasted image 20241008165848.png]]




