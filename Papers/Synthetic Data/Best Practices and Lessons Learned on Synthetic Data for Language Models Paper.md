April 11, 2024
[[DeepMind|Google DeepMind]], Stanford, Georgia Tech
[Best Practices and Lessons Learned on Synthetic Data for Language Models](https://arxiv.org/abs/2404.07503)
#zotero 
Takeaway: Nothing particularly interesting in this paper. Added one or two to vocab/reading list. It's all stuff that we know!


-----

- Motivation
	- Synthetic data is a promising solution to address the challenges of data scarcity, privacy concerns, and the sheer cost of collecting and annotating data. The hope is that with synthetic data we can overcome the limitations of real-world data to develop more *robust, reliable, and fair AI models*.
	- Can be created through algorithms, generative models, or even simulations.
	- Synthetic data can be generated at scale, providing an abundant supply of training and testing data for AI models. This is particularly valuable in domains where real-world data is scarce or difficult to obtain. 
	- Can be tailored to specific requirements, like ensuring balanced representation of different classes by introducing controlled variations (like up-weighting low-resourced languages).
	- Can help mitigate privacy concerns by creating anonymized or de-identified data without sensitive personal information (relevancy in healthcare).
- Challenges
	- Factuality and fidelity of synthetic data; models trained on false/hallucinated/biased synthetic data may fail to generalize to real-world situations.
- Synthetic Data in Training
	- Reasoning
		- Mathematical Reasoning
			- [[Minerva]] (2022) and [[DeepSeekMath]] (2024) trained on math-targeted pre-training data
			- [[WizardMath]] (2023) leveraged a series of operations to increase the complexity of questions and answers using GPT-3.5, and [[MetaMath]] (2023) rewrote questions from [[MATH]] and [[GSM8K]] by rewriting them in different ways, like semantic rephrasing, self-verification, and backward reasoning.
			- GAIR-Abel found that the format of the augmented answers is crucial final performance, and that answers beginning with a paraphrase of a question followed by a step-by-step solutions shows better performance than vanilla formatting.
			- Xwin-Math (2024) scaled up synthetic SFT data to 1M examples and found that LLaMA-2 7B could still benefit from more SFT data scaling.
			- MMIQC (2024) composed a bundle of datasets infusing SFT style data with a subset of high-quality math pre-training data.
			- [[AlphaGeometry]] (2024) tried to ensure the correctness of generated synthetic math data by proposing solutions that guide a symbolic deduction engine in verifying the correctness of each branch when solving problems, combining the power of synthetic data generation with a rigorous verification process, achieving gold medal results.
		- Code
			- CodeRL (2022) presents an actor-critic approach to improve pretrained LMs with feedback signals on synthetic code samples.
			- Intercode (2024) is a framework that tries to use Reinforcement Learning to solve the code generation problem.
			- [[Reflexion]] (2024) employs externally or internally simulated linguistic feedback signals to improve code reasoning capabilities of LMs.
			- Code Alpaca creates 20k code instructions using [[Self-Instruct]] and finetunes ChatGPT.
			- [[WizardCoder]] (2023) introduces Code [[Evol-Instruct]] to guide ChatGPT with heuristic prompts to enhance the complexity and diversity of synthetic data.
			- [[Magicoder]] (2023) developed an [[OSS-Instruct]] technique to generate 75k diverse synthetic instruction samples from open-source code snippets.
		- Other reasoning tasks
			- STaR (2022) generates synthetic CoT rationales and filters out those leading to wrong answers.
			- Mind's Eye (2022) trains a text-to-code model with synthetic "text-description -> rendering code" data, and when the model creates rending code, i's executed in a physical engine (Deepmind MuJoCo), and the rendering results are injected into the context, allowing even small LMs with Mind's Eye to achieve performance comparable to models 100 times larger.
	- Tool use and Planning
		- Learning tool-use through synthetic trajectories (real-world tool-using data collection is time-consuming and the distribution of calls might be skewed)
			- LaMBDA (2022) was trained not only on web documents but also on interaction data between crowdworkers and he model itself, with the synthetic data annotated with calls to appropriate tools.
			- [[Toolformer]] (2024) learns to decide which APIs to call and what arguments to pass by training on template-generated data.
			- [[Galactica]] (2022) infused API-calling data into pre-training mixture.
			- ToolAlpaca (2023) is a novel framework to automatically generate a diverse tool-use corpus, building a multi-agent simulation environment and letting agents select and use tools iteratively.
		- Learning to plan in synthetic environments (decomposing complex tasks into subtasks and finishing the subtasks in a reward-optimal way).
			- Inner Monologue (2022) leverages natural language from feedback generated by the environment to teach LLM-baesd robots planning. Such feedback significantly improves high-level  instruction completion.
			- VIMA (2022) creates a multi-modality simulated environment called VIMA-Bench supporting extensible collections of objects and textures, to help compose a large number of realistic planning tasks ("Rerrange objects on the table to match a given scene")
			- [[Voyager]] (2023) deploys a number of GPT-4-based agents to interact with the synthetic Minecraft environment, and finds that the agents can unlock new skills faster and complete planning more efficiently using synthetic feedback.
	- Multimodality
		- Reverse rendering from vision to text 
			- Web-scraped image-text pairs are usually noisy and only have coarse-grained correspondence. We can use data synthesis pipelines built with image rendering engines!
			- Pix2Struct (2023) renders HTML code into website screenshots, and challenges itself to derender a masked screenshot to full HTML code.
			- MatCha (2023) and DePlot (2023) render tabular data into charts with Python plotting libraries and pretrain a foundation model to, given a rendered image, produce the code and/or tabular data.
			- Borkman et al (2021) propose using game engines (eg Unity) as the synthetic data generator to help CV research.
		- Multimodal instruction-following
			- Long-form instruction-response pairs requiring reasoning and instruction-following capabilities is expensive for humans to create.
			- [[LLaVA]] uses existing image captions to prompt GPT-4 (in text-only mode) to write diverse and long-form prompt-answer pairs.
	- Multilingual
		- Back-translation augmentation
			- Many multilingual language models use [[Back-Translation]] as a data augmentations method, creating synthetic parallel training data from monolingual data sources.
			- Researcher have explored different sampling methods for backtranslation (eg [[Beam Search]], [[Constrained Decoding]], etc.) 
		- Generating multilingual questions and answers at scale
			- Generation and utilization of synthetic multilingual question-answer pairs to improve LM cross-lingual question answering.
			- Some translate existing monolingual questions/answers into other languages.
			- Another involves using *Question Generation* (QG) models to produce synthetic questions in a cross-lingual fashion based on answers and/or source texts.
	- Alignment
		- Instruction Following
			- [[Self-Instruct]] (2022) and [[Alpaca]] (2023) are both using LLMs to generate instruction-following data covering a wide range of scenarios. They pick a small subset of "seed instruction following samples" and then ask the LLM to imitate the format to generate more demonstrations.
				- A problem with this method is how to keep the generated data high quality.
			- [[Evol-Instruct]] (2023) adds complexity to simple instructions via prompting.
			- [[Orca]] leverages LLMs to revise the instructions and responses iteratively to include high-quality explanation traces, and find that trained models have improved performance in many NLP tasks.
			- [[UltraChat]] is a large-scale and multi-round synthetic dialogue dataset, generated by two separate ChatGPT Turbo API models (one as user, one as assistant).
			- Sycophancy is an important problem that sometimes falls out of instruction-following training. Wei et all (2023) generate synthetic data to encourage models to be robust to user opinions and adds this data in a finetuning step.
		- Mitigating Hallucination
			- There are many methods of generating synthetic SFT data that can improve capabilities like reasoning and alignment...
			- Jones et al (2023) designed a synthetic task where hallucinations can be readily evaluated.
			- Tian et al (2023) uses automated fact-checking and confidence scores to rank factuality scores of responses, and then uses this to finetune language models with DPO to improve their factuality.
			- Continued research in this area is limited by the lack of synthetic tasks for which hallucinations can be scalably evaluated.
		- Aligning with shard human preference and values
			- Directly finetuning on value-aligned or human-preferred data is straightforward method for aligning language models, but this data can be expensive to collect.
			- [[Reinforcement Learning from Human Feedback|RLHF]] (2017, 2018, 2022) involves training a reward model with human data to act as the proxy of human judgement, which guides the optimization of the LM generation policy.
			- [[Constitutional AI|CAI]] proposes to use a small set of principles to steer AI-generated critiques and feedback.
			- Alignment with human values is a wide subject, and papers use synthetic datasets to simulate a wide variety of scenarios involving:
				- Ethical dilemmas (Perez et al, 2022)
				- Social interactions (Liu et al, 2023)
				- Cultural norms (Ziems et al, 2023)
			- Researchers must continuously refine and improve the quality and diversity of synthetic data and incorporate more complex and comprehensive scenarios that better capture the intricacies of human values and preferences.
- Synthetic data in Evaluation
	- Factuality
		- Factuality evaluation aims to ensure consistency of knowledge in the AI system's output with the knowledge provided by its training data nd knowledge base.
		- Model-based approaches for detecting hallucination are more robust than statistical methods (eg relying on n-grams to calculate overlaps). 
	- Safety
		- Read-teaming is a powerful technique for evaluating safety and robustness. Perez et al 2023 use LMs to generate datasets to evaluate the behavior of other LLMs.
		- Hubinger et al 2024 leverage synthetic data to trigger backdoor attacks 
	- Assisting human evaluation
		- Lots of GPT-4 as judge papers ([[AlpacaEval]], [[MT-Bench]])
		- Code Mind and CRUXEval are two examples of similar judges for coding tasks, hoping to show strong correlation with real human judgements.
- Challenges
	- Misuse of synthetic data might proliferate misinformation
	- Synthetic data might cause ambiguity in AI alignment; CAI can introduce significant ambiguity and uncertainty. Synthetic data may not represent the nuances and complexities of human values and preferences.
	- Training with synthetic data makes decontamination harder.
- Directions for future work
	- Impressive recent performance of many small, "overtrained" language models like the [[Mistral]] models, [[Gemma]] models demonstrate the necessity of training with large amounts of tokens (even passing chinchilla-optimality).
	- Further research should investigate scaling laws for synthetic data, and determine optimal balance between quantity and quality of synthetic samples.
	- Further research should focus on developing new advanced techniques (or ones based on existing ones like [[Generative Adversarial Network|GAN]] or [[Diffusion Model]] ) that can control and manipulate specific attributes of the generated data, enabling the creation of diverse and customizable synthetic datasets.
	- Incorporation of [[Retrieval-Augmented Generation|RAG]] to make sure that methods include domain-specific knowledge and ensure that data adheres to the underlying patterns present in the target domain.
	- Future research can explore the use of synthetic data for high-fidelity [[Scalable Oversight]]. 
	- Emergent self-improvement is an interesting idea -- can a model generate synthetic data that is *better* than the data it was trained on, enabling it to improve itself? Can it bootstrap its own performance?
		- eg [[Self-Play Fine-Tuning|SPIN]], Weak-to-Strong Generalization (2023), Large Language Models can Self-Improve (2022), [[Self-Reward]]ing langauge models (2024)
		- The upper bound of self-improvement and the underlying reasons for it remain open questions.


Abstract
> The success of AI models relies on the availability of large, diverse, and high-quality datasets, which can be challenging to obtain due to ==data scarcity, privacy concerns, and high costs==. Synthetic data has emerged as a promising solution by generating artificial data that mimics real-world patterns. This paper provides an ==overview of synthetic data research, discussing its applications, challenges, and future directions==. We present empirical evidence from prior art to demonstrate its effectiveness and highlight the importance of ensuring its factuality, fidelity, and unbiasedness. We emphasize the need for responsible use of synthetic data to build more powerful, inclusive, and trustworthy language models.