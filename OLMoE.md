September 3, 2024
[[Allen Institute|AI2]] (Muenninghoff et al.)
[OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
MoE model from [[Allen Institute|AI2]]

Abstract
> We introduce OLMoE, a fully open, state-of-the-art language model leveraging sparse Mixture-of-Experts (MoE). OLMoE-1B-7B has 7 billion (B) parameters but uses only 1B per input token. We pretrain it on 5 trillion tokens and further adapt it to create OLMoE-1B-7B-Instruct. Our models outperform all available models with similar active parameters, even surpassing larger ones like Llama2-13B-Chat and DeepSeekMoE-16B. We present various experiments on MoE training, analyze routing in our model showing high specialization, and open-source all aspects of our work: model weights, training data, code, and logs.



![[Pasted image 20241008165118.png|500]]
Pareto curve of cost to performance for small models, at the time

![[Pasted image 20241008165123.png|400]]
Makes the comparison to larger models more favorable

![[Pasted image 20241008165157.png|500]]
For both KTO and DPO.... we're starting to see small models respond to fine-tuning.

