---
aliases:
  - Synthetic Data Generation
  - SDG
---

References:
- Eugene Yan's [Synthetic Data blogpost](https://eugeneyan.com/writing/synthetic/)

Related: [[Data Augmentation]]

Relative to human annotation, it's faster and cheaper to generate task-specific synthetic data. The quality and diversity of synthetic data can often exceed that of human annotators, leading to improved performance and generalization when models are finetuned on synthetic data. It also lets us synthetic data sidesteps privacy and copyright concerns by avoiding reliance on user data or possibly copyrighted content.

Synthetic data is a promising solution to address the challenges of data scarcity, privacy concerns, and the sheer cost of collecting and annotating data. The hope is that with synthetic data we can overcome the limitations of real-world data to develop more *robust, reliable, and fair AI models*.
- Can be created through algorithms, generative models, or even simulations.

There are two main approaches to generate synthetic data:
1. [[Distillation]] from a stronger model
	- Distillation transfers knowledge and reasoning skills from a stronger teacher to a weaker, but more efficient student, optimizing for response quality and computation efficiency.
2. [[Self-Improvement]] on the model's own output
	- When the model learns from its own responses via an iterative loop. It avoids external dependencies and contractual restrictions. Nonetheless, it limits learning to the model's initial abilities and can amplify biases and errors.

![[Pasted image 20240525170708.png]]
Above: [[TinyStories]], [[Phi-1]], [[Unnatural Instructions]], [[Alpaca]], [[AlpaGasus]], [[Vicuna]], [[Orca]], [[Orca 2]], [[WizardLM]], [[WizardCoder]], [[Magicoder]], [[WaveCoder]], [[Starling]], [[AlphaGeometry]], [[Web Rephrase Augmented Pre-training|WRAP]], [[Self-Instruct]], [[Self-Play Fine-Tuning]], [[Instruction Backtranslation]], [[Reinforced Self-Training|ReSTEM]], [[Constitutional AI|CAI]], [[Self-Reward]]

Important considerations when generating synthetic data:
- Quality of content
- Diversity of content
- Complexity/Depth/Nuance/Difficulty of content
- Number of turns
- Average turn length
- (Sometimes) Removing undesirable behaviors (eg "as an AI language model")

Examples
- Distillation
	- [[Unnatural Instructions]] (Dec 19, 2022): Generates synthetic data from GPT-3.5, then uses synthetic data to finetune T5-LM. Starting with 15 seed human-written examples, they prompt GPT-3.5 with three examples to generate a fourth synthetic example (instruction+context), using nucleus sampling for creativity when generating prompts, and greedy decoding for accuracy when generating responses. Authors filtered instruction-input pairs that were low-quality, and included a template expansion step to increase format diversity.
	- [[Alpaca]] (Mar 13, 2023): Finetuned LLaMa-7B on 52k instruction-following examples from GPT-3.5; used the same 175 human-written instruction-response pairs from the Self-Instruct seed set, and generated more pairs from GPT-3.5 via few-shot prompting (a la Self-Instruct).
	- [[Vicuna]] (Mar 30, 2023): Finetuned LLaMA-7B and LLAMA-13B on a filtered subset of 125k user conversations from [[ShareGPT]].
	- [[WizardLM]] (Apr 24, 2023): Generated more complex and diverse instructions and responses using GPT-3.5. Distinguishes between *in-depth evolutions* and *in-breadth evolutions*, with the former making instructions more complex via five types of prompts, such as "adding constraints, deepening, increased reasoning steps," the latter increasing topic coverage, skill coverage, and overall dataset diversity.
	- [[Orca]] (Jun 5, 2023): Explored how similar models can imitate the reasoning process of larger, stronger models via training on explanation traces. Asked GPT-4 and GPT-3.5 to generate response with traces, using system instructions like "ELI5," "think step-by-step," and "justify your response," and then finetuned LLaMA-13B on the results.
	-  [[WizardCoder]] (Jun 14, 2023): Applies the [[Evol-Instruct]] method from WizardLM to make code instructions more complex, enhancing fine-tuning. Starts with a 20k instruction-following dataset from Code Alpaca, and applied Evol-Instruct, then finetuned StarCoder-15B on the resulting data.
	- [[Phi-1]] (June 20, 2023): Trained a 1.3B param model to write simple Python functions via docstrings; trained on "textbook-quality" data filtered from the Stack and StackOverflow via a Random Forest quality classifier trained/bootstrapped off of 100k GPT-4 annotations.
	- [[Phi-1.5]] (Sep 11, 2023): Goes further by generating 20B tokens of synthetic textbooks on common sense reasoning and general knowledge of the world.
	- [[Orca 2]] (Nov 18, 2023): Continues down the path of reasoning by finetuning LLaMA-7B/13B to use various reasoning strategies for different tasks. Distilled a synthetic dataset of 817k samples from GPT-4 with various reasoning techniques like step-by-step, recall-then-generate, recall-reason-generate. These synthetic samples included information on how to determine the most effective reasoning technique for each task.
	- [[Starling]] (Nov ?, 2023): Incorporates preference tuning on a GPT-4-labeled ranking dataset, consisting of 183 chat prompts each having 7 responses distilled from popular models like GPT-4, GPT-3.5-turbo, Mistral-7B-Instruct, and LLaMA2-7B, resulting in 3.8M pairwise comparisons (dataset name: Nectar) that are used to finetune a reward model.
	- [[Magicoder]] (Dec 4, 2023): Generates synthetic coding problems by providing example code snippets from open source projects as part of the few-shot prompt, using GPT-3.5 to generate new coding problems. 
	- [[WaveCoder]] (Dec 20, 2023): Extends code instruction-tuning by classifying instruction data into four code tasks: summarization, generation, translation, and repair. They use GPT-4 to generate instructions and code requirements, and GPT-3.5 to generate context and responses, then used GPT-4 to classify the generated samples as good or bad.
- Self-Improvement
	- [[Constitutional AI]] (Dec 15, 2022): Focuses on self-improvement via a list of 16 principles, using an LLM-as-a-Judge paradigm. For instruction tuning, they first generate synthetic responses from the model, then generate self-critiques and revised responses based on the constitution before finetuning the model on the revised responses. For preference-tuning, they sample from the previous instruction-tuned model, and use a LM as a reward model to evaluate which sample is better (based on the constitution), before finetuning the instruction model against the preference model.
	- [[Self-Instruct]] (Dec 20, 2022): Improves instruction-following ability of a LLaMA model by bootstrapping off its own generations. Starting with 175 seed instructions, they first generate instructions, input context, then responses from the model. Then they filter invalid or similar examples before using the remaining 52k instructions/82k I/O pairs to finetune the original model.
	- [[Instruction Backtranslation]] (Aug 11, 2023): Turns generating synthetic data on its head; instead of generating responses for human instructions, they generate instructions for human-written text from the internet using a finetuned "backward" model. Has a self-augmentation (generating instructions for unlabeled responses), and a self-curation step (selecting the high-quality instruction-response pairs), which are both performed iteratively, where augmented and curated data from a previous step is used to finetune an improved model, which is then used to rescore the augmented examples for quality. Resulting model is called Humpback.
	- [[Reinforced Self-Training|ReSTEM]] (Dec 11, 2023): Proposes a two-step, self-training method inspired by expectation maximization. In the E-step, they generate multiple output samples for each instruction, then filter the generations to create synthetic data. In the M-step, they instruction-tuned on the synthetic data from the E-step. The finetuned model is then used in the next E-step.
	- [[Self-Play Fine-Tuning]] (SPIN) (Jan 2, 2024): Proposes a GAN-like approach to instruction tuning, where the main player (discriminator) distinguishes between responses generated by an LLM versus a human while the opponent (generator) generates responses, attempting to be indistinguishable from a human. During iterative self-play, the opponent from the previous iteration $t$ is used to train the main player at $t+1$. 
	- [[Self-Reward]]ing language models (Jan 18, 2024): Applies LM-as-a-Judge on the model so it provides its *own* reward. The goal is to overcome the bottleneck of human preferences as well as allow reward models to learn as the language model improves while finetuning. The goal is to train a model that can both *return helpful/harmless responses, given an instruction* AND *evaluate synthetic instruction-response pairs*. They iteratively finetune a series of models, with each $t$ using augmented training data from the previous $t-1$ model.
- Synthetic Data for Pretraining
	- [[AlphaGeometry]] (Jan 17, 2024): Generated synthetic data to pretrain and finetune a model that can solve geometry problems near the level of a Math Olympiad gold medalist. Uses a symbolic deduction engine to generate derivations of a random set of geometry theorem premises, and identify true statements via forward inference rules. The pretrained a transformer on all 100M synthetically generated proofs. During proof search, the LM and the symbolic deduction model take turns; at each turn, the LM is provided with a problem statement and generates a construct conditioned on the problem statement and past constructs;  then the symbolic engine is provided with the new constructs to potentially reach the conclusion.
	- [[Web Rephrase Augmented Pre-training|WRAP]] (Jan 29, 2024): A method to augment an existing dataset with synthetic data. Used Mistral-7B-Instruct to rephrase documents in C$, using four different rephrasing styles: "Easy text that children can understand, medium text that's high-quality and Wikipedia-like, hard text that is terse and abstruse, and Q&A text in conventional question-answering format." Trained Transformers on a blend of real and synthetic rephrased data, and sped up pre-training by 3x.

![[Pasted image 20240627201536.png]]
"Four Kinds of Synthetic Data" from [John David Pressman](https://x.com/jd_pressman/status/1797396190210015716)

![[Pasted image 20240710234731.png]]


![[Pasted image 20240712174504.png|300]]
This is from the AgentInstruct/Orca3 paper
T