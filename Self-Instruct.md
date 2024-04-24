Dec 20, 2022 -- Various universities, including [[Allen Institute]]
Paper: [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)

- Similar to [[Unnatural Instructions]], [[Self-Instruct]] consists of 82k examples automatically generated using InstructGPT conditioned on a set of *seed task examples* (175 tasks in total; one example per task; 8 examples are sampled for in-context learning).
- Self-Instruct decouples the example generation by ==first generating the instruction==, ==then the input (conditioned on instruction),== and ==then the output==.
- For classification tasks, authors choose to instead first generate the possible output labels, and then condition the input generation on each class label, to avoid biasing towards a specific label. 
- Although the generated instructions are mostly valid, the generated outputs are often noisy.

Related/Improved: [[Evol-Instruct]]

Note: *Self-Instruct* refers to the technique/pipeline, but it's also often used as a descriptor for datasets that are created with this technique -- eg "XYZ, a Self-Instruct dataset".


Paper Abstract:
> Large "instruction-tuned" language models (i.e., finetuned to respond to instructions) have demonstrated a remarkable ability to generalize zero-shot to new tasks. Nevertheless, they depend heavily on human-written instruction data that is often limited in quantity, diversity, and creativity, therefore hindering the generality of the tuned model. We introduce Self-Instruct, a ==framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations==. Our ==pipeline generates instructions, input, and output samples from a language model, then filters invalid or similar ones before using them to finetune the original model==. Applying our method to the vanilla GPT3, we demonstrate a 33% absolute improvement over the original model on Super-NaturalInstructions, on par with the performance of InstructGPT-001, which was trained with private user data and human annotations. For further evaluation, we curate a set of expert-written instructions for novel tasks, and show through human evaluation that tuning ==GPT3 with Self-Instruct outperforms using existing public instruction datasets by a large margin==, leaving only a 5% absolute gap behind InstructGPT-001. Self-Instruct provides an almost annotation-free method for aligning pre-trained language models with instructions, and we release our large synthetic dataset to facilitate future studies on instruction tuning.

![[Pasted image 20240415233101.png]]

![[Pasted image 20240418164722.png]]
"How do we expand our data without getting more humans in the loop?"
- Start with high-quality, human prompts. Ask a strong LM to create a list of similar, but still diverse, prompts.
- Once you have a list of prompts, use ChatGPT or another model to generate completions -- then you have a very big list of Q/A pairs, but you don't need to go through the bottleneck of having humans sit down and write down either.


![[Pasted image 20240424005031.png]]