#article 
Link: https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives

-----

[[Reinforcement Learning from Human Feedback]] is an integral part of the modern LLM training pipeline due to its ability to incorporate human preferences into the optimization landscape, which can improve the model's helpfulness and safety.

Let's break down RLHF in a step-by-step manner

Table of contents:
1. The canonical LLM training pipeline
2. Reinforcement Learning with Human Feedback (RLHF)
3. RLHF in Llama 2
4. RLHF alternatives

----
# (1/4) The Canonical LLM Training Pipeline
Modern transformer-based LLMs often times undergo a 3-step training process:
- Pretraining
- Instruction-tuning using [[Supervised Fine-Tuning]] (SFT)
- Alignment using [[Reinforcement Learning from Human Feedback]] (RLHF)

Pretraining
- Typically occurs on a vast text corpus comprising billions or trillions of tokens.
- We employ a straightforward language modeling objective of next-token prediction, and use a [[Semi-Supervised Learning]] paradigm to sort of bootstrap training data from a large internet text corpus of unlabeled text data -- with SSL, the labels are inherent in the data itself.

Supervised Finetuning
- Involve another round of next-token-prediction, but now we work with *==instruction-output pairs==*, as depicted below:
![[Pasted image 20240409132058.png|450]]
- The instruction is the input given to the model (sometimes accompanied by an optional text input, depending on the task). The output represents a *desired response* similar to what we expect the model to produce.
- Supervised Finetuning uses a (typically) much smaller dataset than pretraining; this is because it requires instruction-output pairs rather than just raw text. To compile such a dataset, a human must write the desired output given a specific instruction -- ==creating such a dataset is a lot of work!==

Finally, we have an "Alignment" step, where the objective is to align the outputs of the language model with human preferences.
- Following the InstructGPT example
![[Pasted image 20240409133332.png|450]]


# (2/4) Reinforcement Learning with Human Feedback (RLHF) 
- The RLHF pipeline takes a pretrained model and finetunes it in a supervised fashion (step 2), and then further aligns it with human preferences using [[Proximal Policy Optimization]] (PPO).
- Let's look at the RLHF pipeline as a series of three separate steps:
	1. Supervised finetuning of the pretrained model
	2. Creating a reward model
	3. Finetuning via proximal policy optimization

In RLHF Step 1:
- We create or sample prompts (e.g. from a database) and ask humans to write high-quality responses.
- We then use this dataset to finetune the pretrained base model in a supervised fashion.

In RLHF Step 2:
- We use this model produced by supervised finetuning to create a *reward model*
	- For each prompt, we generate 4-9 responses from our finetuned LLM created in the prior step.
	- A ==***human*** then ranks these responses based on their preference==.
		- Although this ranking process is time-consuming, it might be somewhat less labor-intensive than creating the dataset for supervised finetuning. ==This is because ranking responses is likely simpler than writing them.==
	- Upon compiling a dataset with these rankings, ==we can design a reward model that outputs a reward score for the optimization subsequent stage in RLHF==.
		- This reward model generally originates from a checkpoint of the LLM created in the prior supervised finetuning step. We replace its classification head with a regression layer, featuring a single output node.


In RLHF Step 3:
- We use the reward model to finetune the previous model obtained from SFT.
![[Pasted image 20240409134518.png|450]]
- We are now updating the SFT model using proximal policy optimization ([[Proximal Policy Optimization|PPO]]) based on the reward scores from the reward model cerated in step 2.
- More details on PPO are out of scope of this article.
	- ((😡))
# (3/4) RLHF in Llama 2
- In the previous section, we looked at the RLHF procedure described in OpenAI's InstructGPT paper -- but how does this process compare to that used by Meta for the [[LLaMA 2]] model?
- Meta AI utilized RLHF in creating the Llama-2-chat models as well! But there are several distinctions between this approach and that originally used in the InstructGPT paper.
![[Pasted image 20240409134717.png]]

Llama-2-chat follows the same SFT step on instruction data as InstructGPT used in *RLHF Step 1*

In *RLHF Step 2,* Meta created *==two reward models, instead of one!==* ((I believe one for Helpfulness and one for Harmlessness))
Furthermore, the Llama-2-chat model ==evolves through multiple stages, with reward models being updated based on the emerging errors== from the Llama-2-chat model. There's also an added ==rejection sampling== step.

### Margin Loss
- Another distinction not depiction in the above-mentioned annotated figure relates to ==how model responses are ranked to generate the reward model==.
	- In the standard InstructGPT approach for RLHF PPO discussed previously, the researchers collect responses that rank 4-9 outputs, from which the researchers create "k choose 2" comparisons.
		- For example, a human labeler ranking four responses (A-D) like A < C < D < B yields 6 comparisons
			- `A<C , A<D, A<B, C<D, C<B, D<B`
	- Similarly, ==Llama2's dataset is based on binary comparisons of responses, but it appears that each human labeler was only presented 2 responses, rather than 4-9 responses, per labeling round.==
		- The idea is that rankings will be more accurate if done head-to-head, instead of trying to rank (eg) 9 responses in a single ranking.
	- Alongside each binary rank, a =="margin" label== (ranging from "significantly better" to "negligibly better") is gathered, which ==can optionally be used in the binary ranking loss via an additional margin parameter to calculate the gap between two responses==.

### Two Reward Models
- As mentioned before, two reward models in Llama 2 are used, instead of one! One reward model is based on *helpfulness*, and the other on *safety* (harmlessness).
	- The final reward function is a linear combination of these two scores.
![[Pasted image 20240409140735.png|400]]

### Rejection sampling
- The authors of Llama 2 employ a training pipeline that iteratively produces *multiple* RLHF models (from RLHLF-V1 to RLHF-V5)
- ==Instead of relying solely on the PPO method, they employ TWO algorithms for RLHF finetuning!==
	1. PPO
	2. [[Rejection Sampling]]

In ==Rejection Sampling==, K outputs are drawn, with the one with the *highest reward* being chosen for the gradient update during the optimization step:
![[Pasted image 20240409140919.png]]
Rejection Sampling serves to select samples that have a high reward score in each iteration.
==As a result, the model undergoes finetuning with samples of a higher reward compared to PPO, which updates based on a single sample each time.==
- ((But what is the effect of doing this on the model? Do we benefit from learning from our worst generations too?))

After the initial phases of supervised finetuning, models are exclusively trained using rejection sampling, before later combining both rejection sampling and PPO.
# (4/4) RLHF Alternatives
- Now that we've discussed and defined the RLHF process, a pretty elaborate procedure, one might wonder whether it's even worth the trouble!

### Constitutional AI (Dec 2022, Anthropic) [[Constitutional AI|CAI]]
- In this Constitutional AI paper, researchers propose a self-training mechanism based on a list of rules that humans provide.
- Similar to the InstructGPT paper earlier, the proposed method uses an RL approach

![[Pasted image 20240409143053.png|450]]

### Wisdom of Hindsight makes LMs better instruction followers (Feb 2023)
- Shows supervised approaches to LLM finetuning can indeed work well; researchers propose a relabeling-based supervised approach for finetuning that outperforms RLHF on 12 BigBench  tasks.
- In a nutshell, ==Hindsight Instruction Labeling== (HIR) works in two steps:
	1. Sampling
		- Prompts and instructions are fed to the LLM to collect responses
		- Based on an alignment score, the instruction is relabeled where appropriate in the training phase
	2. Training
		- The relabeled instructions and the original prompts are used to finetune the LLM.

==Using this relabeled approach, the researchers turn failure cases (where the LLM doesn't generate output that matches the original instructions) into useful training data for supervised learning!==


### Direct Preference Optimization: Your Language Model is Secretly a Reward Model (May 2023)
- [[Direct Preference Optimization|DPO]]
- Direct Preference Optimization is an alternative to RLHF with PPO where the ==researchers show that the cross entropy loss for fitting the reward model in RLHF can be used to directly finetune the LLM==.

### Contrastive Preference Learning: Learning from Human Feedback without RL (Oct 2023)
- Like DPO, Contrastive Preference Learning (CPL) is an approach to simplify RLHF by eliminating the reward model learning.
- Like DPO, CPL uses a supervised learning objective -- specifically, a contrastive loss
- Applied in a robotics environment, but CPL could also be applied to LLM finetuning.

### Reinforced Self-Training (ReST) for Language Modeling (Aug 2023)
- Uses a sampling approach to create an improved dataset, ==iteratively training on increasingly higher-quality subsets to refine its reward function==.
- "Achieves greater efficiency compared to standard online RLHF methods (eg PPO) by generating its training dataset offline."
	- Unfortunately, a comprehensive comparison to standard RLHF PPO methods as used in InstructGPT or Llama2 is missing.

### [[Reinforcement Learning from from AI Feedback|RLAIF]]: Scaling Reinforcement Learning from Human Feedback with AI Feedback (Sep 2023)
- The recent RLAIf study shows that rating for the ==reward model training in RLHF don't necessarily have to be provided by humans, but can be generated by an ==LLM== (here: PaLM 2). 
- ==Human evaluators prefer the RLAIF model half the time over traditional RLHF models, meaning they don't actually prefer one model over the other.==

The outcome of this study is basically that we might be able to make RLHF-based training more efficient and accessible -- but it remains to be seen how these RLAIF models perform in qualitative studies that focus on safety and truthfulness of the information content, which is only partially captured by human pereference studies.











