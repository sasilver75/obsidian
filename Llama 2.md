
July 18, 2023 (~5 months after [[LLaMA]])
[[Meta AI Research]]
Paper: [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
#zotero 
Takeaway: A great new open-weights model from Meta, available in a variety of sizes in either base or chat variants. Authors give a lot of time to the RLHF and safety portions of the paper.

> "LLaMA-2-chat-70B was known to be a very disliked model." 
> - Nate Lambert, "Frontiers in Synthetic Data" article
>  "LLaMA 1 and LLAMa 2 were quite mid, but then LLaMA 3 got pretty good."
>  - Yi Tay, Chief Scientist @ Reka, Latent Space interview

----

Technical details:
- Family: 7B, 13B, 70B (including both a base model and a chat model at every level)
- Trained on: 2T tokens (with the most factual data sources upsampled), increased from 1.4T tokens in LLaMA 1
- Context length: 4096 tokens, increased from 2048 in LLaMA 1
- RLHF with [[Proximal Policy Optimization|PPO]], [[Rejection Sampling]]
	- Samples $k$ responses from the LLM for each prompt, scores each response using the reward model, selects the bets response, and fine-tunes (in a supervised fashion) on this example. Generating multiple samples can drastically increase the maximum reward of samples observed during fine-tuning.
- "Ghost Attention"
	- A method of improving prompt adherence over multi-turn interactions.
	- Given a dialogue session, GAtt samples an instruction that *should be followed*in a conversation, concatenates this instruction to every user message in the conversation, resamples responses to each user message, *removes* evidence of those instructions from user messages, and fine-tunes the model on the multi-turn dialogue with SFT.
- Use of [[Grouped Query Attention]], a modified version of multi-headed causal self attention where we divide the N total self-attention heads into groups that share key and value heads. It's an interpolation between vanilla multi-headed self attention and [[Multi-Query Attention]] that is the best of both worlds.
- Use of [[SwiGLU]] activations, rather than [[Rectified Linear Unit|ReLU]] activations. This activation function requires three matrix multiples (more expensive than a ReLU), but has been found to yield improvements in performance relative to other activation functions.
- Use of [[Rotary Positional Embedding]] (RoPE), rather than absolute or relative positional embeddings. RoPE finds a balance between the absolute and relative position of each token in a sequence by encoding absolute position with a rotation matrix, and adding relative position information directly into the self-attention operation. Good for longer sequences.
- Use of [[RMSNorm]] normalization, a simplified version of [[Layer Normalization|LayerNorm]] that has been shown to improve training stability and generalization. Adopts a pre-normalization variant of RMSNorm, meaning that normalization is applied *prior* to the major layers in the transformer block, rather than after.
- Use of [[AdamW]] optimizer, with a cosine learning rate schedule, weight decay, and gradient clipping.
- Uses the same tokenizer as LLaMA 1, employing [[Byte-Pair Encoding]] (BPE), using the implementation from [[SentencePiece]].
	- We split all numbers into individual digits and use bytes to decompose unknown UTF-8 characters, using a 32k token vocabulary size.
- Commercial use license
- [[LLaMA Guard]]

Notes:
- With respect to the Co2 emissions, the 70B model emitted 291.42 tCO2eq; compare with a round-trip emission of SFO->JFK->SFO airplane flight being 1.5 tCO2eq *per-passenger*.
- They bootstrap their SFT stage with publicly available instruction-tuning data used for the [[FLAN-T5]] model (though it sounds that they filtered some out, and used other third-party datasets and their own vendor-based annotation efforts).
	- They found that having SFT annotations on the order of "*tens of thousands*" is enough to achieve a high-quality result (though they didn't quantify this).
- When collecting human preference data, human annotators select which of two model outputs they prefer (a binary comparison protocol), because it enables us to maximize the diversity of collected prompts ((?)), but they note that there are other strategies worth considering.
- Annotation procedure
	- We ask annotators to first write a prompt, then choose between two sampled model responses, based on provided criteria. 
		- To maximize diversity, the two responses to a given prompt are sampled from two different model variants, and varying the temperature hyperparameters.
	- In addition to requiring a binary preference choice, the annotators also have to say whether their choice is *significantly better, better, slightly better, or negligibly better/unsure*.
	- Annotators also focus on helpfulness and safety, which (I think?) are given human preferences in separate stages, allowing the annotators to focus on adversarial prompts, among other guidance.
		- During the safety stage, we additionally collect a safety label that bins model responses into one of three categories:
			1. The preferred response is safe, and the other response is not
			2. Both responses are safe
			3. Both responses are unsafe.
		- 18%, 47%, 35% of the safety dataset fell into each bin, respectively.
	- Human annotations were collected in batches on a weekly basis. As we collect more preference data, our *reward models* improved, and we were able to train progressively better versions of LLaMA 2-Chat.
	- We end up collecting a dataset of ==over 1 million binary comparisons== based on humans applying out specified guidelines.
- Reward Modeling
	- The reward model takes a model response and the corresponding prompt (including contexts from previous turns) and outputs a *scalar score* to indicate the quality of the generation.
	- Because sometimes Helpfulness and Safety pull in opposite directions, we train ==two separate reward models==, with one optimized for helpfulness and another for safety.
	- Reward models are initialize from pretrained chat model checkpoints, ensuring that both models benefit from knowledge acquired in pretraining, and preventing cases where two models would have an information mismatch, which could result in favoring hallucinations. The only change is that the classification head for NTP is replaced with a regression head for outputting a scalar reward.
	- Reward models are trained with the following objective:
		- ![[Pasted image 20240504120721.png]]
- Meta combines their newly collected human preference data with existing open-source preference data to form a larger dataset, and they don't observe any negative transfer from the open-source preference datasets.
- RL Preference tuning
	- Authors explore both [[Proximal Policy Optimization|PPO]] and [[Rejection Sampling]] fine-tuning, where they sample K outputs from the model and select the best with the reward, then use the selected outputs for a gradient update.
	- Comparison
		- Breadth: In Rejection Sampling, the model explores K samples for a given prompt, while only one generation is done for PPO.
		- Depth: In PPO, during training at step *t*, the sample is a function of the updated model policy from t-1 after the gradient update of the previous step. In Rejection Sampling, we sample all outputs given the *initial policy* of our model to collect a new dataset, before the applying fine-tuning similar to SFT... But since the Meta folks perform iterative model updates, the fundamental differences are less pronounced.
		- Until RLHF(V4), we use only Rejection Sampling fine-tuning, and after that they combine the two sequentially, applying PPO *on top of the resulted Rejection Sampling checkpoint* before sampling again.
- ==[[Rejection Sampling]]==
	- We perform rejection sampling only with our largest 70B LLaMA 2-Chat -- all smaller models are fine-tuned on rejection-sampled data from the larger model, thus distilling the large model capabilities into the smaller ones.
	- At each stage, we sample K answers for each prompt from the most recent model, and score each sample given the best reward model accessible at the time of the experiment, and then select the *best answer* for a given prompt.
	- (See end of Page 14 for description of a problem that occurred when they trained version X of the RLHF model on version X-1 rejection-sampled data ONLY; they ended up training (eg) V4 on the best outputs from *all* of V3,V2,V1.)
	- (Note: I believe the outputs of the rejection sampling process are specifically used for fine-tuning via RLHF with PPO; i.e. we don't do SFT on the results.)
- ==Note:== As we train progressive versions of the model (eg RLHFv2, RLHFv3), we *always start* from the base model on each new RLHF version.
- [[Proximal Policy Optimization|PPO]]
	- We iteratively improve the policy by sampling prompts *p* from our dataset *D* and generations *g* from the policy $\pi$ and use the PPO algorithm and loss function to achieve the following objective:
	- ![[Pasted image 20240504144007.png|200]]
	- Using the following final loss function:
	- ![[Pasted image 20240504144030.png|300]]
		- Note that this loss function contains a penalty term for diverging from the *original* policy $\pi_0$. This constraint is useful for training stability, and to reduce reward hacking resulting in high scores from the reward model but low scores from human evaluators.
		- Above, $R_c$ is a piecewise combination of the safety and helpfulness reward models $R_s$ and $R_h$.
		- Note: They also identify prompts in their dataset that might elicit potentially-unsafe responses, and in those cases prioritize the scores from the safety model. They also find it important to "whiten" the final linear scores
		- ![[Pasted image 20240504144850.png]]
		- Above: ==The logit function is the inverse of the sigmoid (google it)==, and maps probabilities from the interval(0,1) back to real numbers (-inf, inf). Note that ==the outputs of all of the reward models are in the (0,1) range.==
- ==Ghost Attention==
	- We notice that our early RLHF models aren't good at following instructions that should apply over the course of the entire conversation, across many terms (eg "Respond only in Emojis"). Ghost Attention (GAtt) is a simple method inspired by [[Context Distillation]] to help the attention focus in a multi-stage process.
	- ![[Pasted image 20240504151055.png]]
	- My take on the above is they take a real conversation, augment all of of the user-side interactions by prepending the instruction, then regenerate the agent side with a good model, and then *remove* the prepended instruction text. Now you have a conversation where the model seems to magically be following the instruction, which you can then use to finetune your model in a supervised fashion.
- Safety Fine-Tuning
	- The following techniques were used in safety fine-tuning:
		1. Supervised Safety Fine-Tuning: We gather adversarial prompts and safe demonstrations, and then use a SFT process.
		2. Safety RLHF: We use a safety-specific reward model and gather challenging prompts for rejection-sampling fine-tuning and PPO optimization. We ask annotators to write prompts they believe can elicit unsafe behavior, then compare model outputs and select the safer one.
		3. Safety Context Distillation: We refine our RLHF pipeline with [[Context Distillation]], which involves generating safer model responses by *prefixing a prompt with a safety reprompt*, eg "*You are a safe and responsible assistant*", and then fine-tuning the model on the safer responses *without* the reprompt, essentially *distilling* the safety preprompt (context) into the model. We use a target approach allowing our safety reward model to choose whether to use context distillation for each sample.

Abstract
> In this work, we develop and release ==Llama 2==, a ==collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters==. Our fine-tuned LLMs, called ==Llama 2-Chat==, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. ==We provide a detailed description of our approach to fine-tuning and safety improvements== of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs.

### Preference Data Details:
- Collected ==binary comparisons, rather than other fancier feedback==... also information like *"significant better, better, slightly better, neglibgibly better/unsure"*
- Used ==multi-turn preferences==, where model responses are taken from different model checkpoints with varying temperature, to generate diversity between pairs.
- Focus collection on ==helpfulness and safety== (as opposed to *honesty*), using separate guidelines at data collection time for each vendor (safety is often a much more deceptive prompting style).
- The team added additional ==safety metadata== to the collection, showcasing whic h responses are safe from the models at each turn; when this is passed to the modeling phase, they don't include any examples where the chosen response was unsafe, while the other response *was* safe; "we believe safer responses will be better/preferred by humans, in the end."
- They don't detail additional metadata being logged, but it's likely they did, in order to identify potential bugs and data issues (confusing prompts, requiring tools to solve, etc.)
- Deployed ==iterative collection for distribution management==: "Human annotations were collected in batches on a weekly basis; as we collected more preference data, our reward models improved, and we were able to train progressively better versions for LLaMA 2-Chat".


# Paper Figures

![[Pasted image 20240504111516.png]]
Above: LLaMA 2 is meaningfully "safer" than all (?) open-source competitors, and more than all closed-source competitors besides the best frontier models.

![[Pasted image 20240426014123.png]]
Above: The model goes through a "standard" pre-training -> instruction-finetuning -> RLHF.

![[Pasted image 20240504114014.png]]
Above: An example of helpful and "safe" responses used as annotations for SFT (both responses written by humans).

![[Pasted image 20240504120142.png]]
Above: The scale of preference data was impressive, likely costing $8M+ on dat alone, with way more turns than were often available at that time. Note that the Meta dataset (which unfortunately wasn't open-sourced) has more comparisons, more turns, and longer generations.

![[Pasted image 20240504121522.png]]
Above: Note that they seem to have used same-sized reward models for their models. See that larger reward models are better, and that the models improve across "stages" of training. Authors note that =="reward model accuracy is one of the most important proxies for the final performance of LLaMA 2-Chat"==

![[Pasted image 20240504135705.png]]
Above: The authors test both PPO and [[Rejection Sampling]] fine-tuning, where they sample K outputs from the model and select the generation(s) with the best reward, and use the selected output(s) for gradient updates. There's a direct connection between the exploration and maximum reward that we can obtain among the samples; the temperature parameters also plays an important role for exploration, as a higher temperature enables more diverse outputs (see figure 8, below, too).

![[Pasted image 20240504141411.png]]
Above: See the impact of temperature on maximum reward (relevant for rejection sampling, it seems). Note that the optimal temperature is not constant during iterative model improvements; it's therefore optimal to re-adjust the temperature progressively.

![[Pasted image 20240504145207.png]]
Above: Highlighting the contribution of Ghost Attention (GAtt) when it comes to instruction-adherence over multiple turns of conversation.

![[Pasted image 20240504151123.png]]
Above: Highlighting the impact on the attention matrix that the introduction of Ghost Attention (GAtt) had; you can see that it seems to be paying much more attention to the first few tokens in the dialogue, which contain the instruction "Acts as Oscar Wilde."


![[Pasted image 20240426132028.png]]
Above: Showing the improvement on Helpfulness and Harmlessness benchmarks from both human and LM judges as we do multiple rounds of SFT and then RLHFing. See that we appear to be making increasing progress, especially on the harmlessness metric.

![[Pasted image 20240504161943.png]]
Above: Examples of responses before and after safety RLHF

![[Pasted image 20240504162033.png]]
Above: See that increasing the percentage of safety data in model training, the safety score improves meaningfully, but the helpfulness isn't hurt. 
- I'm not sure what it means above for there to be (eg) 100% "safety data pct"; In which phase of training? If there's 100% safety data, does that mean 0% helpfulness data? Which means that helpfulness data doesn't even seem to help helpfulness?

![[Pasted image 20240504162513.png]]
Above: [[Context Distillation]] example

![[Pasted image 20240504162805.png]]
Above: Impact of [[Context Distillation]]


![[Pasted image 20240426021803.png]]
An awesome picture showing how the RLHF process shifts the generated texts towards higher rewards. The X axis is "Reward Model Score".



# Non-Paper Figures
![[Pasted image 20240417224708.png]]