From [[Maxime Labonne]]
https://huggingface.co/blog/mlabonne/agentic-datagen

----

With the consolidation of LLM architectures, the quality of training data has become the most important factor in creating SoTA models. This is true for both pre-training and post-training.

Two innovative approaches have recent emerged to address the challenge of generating high-quality instruction datasets for post-training LLMs:
1. [[AgentInstruct]]
2. [[WizardArena]]/Arena Learning

Both frameworks come from [[Microsoft Research]] and leverage multiple LLMs to create and refine samples.

# AgentInstruct: A Multi-Agent Approach 🛠️
- [[AgentInstruct]] (Mitra et al., 2024) is meant to generate large-scale, diverse, high-quality, and challenging data.
- Uses a pipeline that transforms raw text into refined instructions through multiple stages of processing.
![[Pasted image 20240719000942.png]]
Four main steps:
- ==Seed Collection==: Assemble a diverse collection of raw seeds, such as textbook chapters, web articles, and code snippets. These seeds serve as the foundation for generating new instructions.
- ==Content Transformation==: One or more specialized agents modify each seed into some *intermediate representation that simplifies instruction generation*. These agents perform tasks like generating argument passages, debates, conversations, meeting transcripts, poems, satirical content, etc.
- ==Seed Instruction Generation==: Multiple agents take the transformed seed and generate diverse instructions based on a pre-defined taxonomy of instruction types.
	- So in the domain of reading comprehension, the taxonomy includes 43 question types, ranging from literal comprehension to critical analysis and inference.
- ==Instruction Refinement==: Iteratively enhancing the complexity and quality of the generated instructions. This is achieved through suggester-editor agent pairs, where suggester agents propose ways to increase instruction complexity, while editor agents modify the instructions accordingly.

Each flow in the AgentInstruct pipeline consists of multiple agents powered by LLMs! They can be equipped with tools like search APIs or code interpreters to enhance their capabilities. The roles of each of these agents are carefully designed in system methods to ensure they perform system tasks effectively.

Authors of AgentInstruct implement flows for 17 different skills, from QA to coding to creative wring.

Using this pipeline, ==authors generate approximately 22 million instructions,== and combine this with 3.8M instructions from other datasets to create a 25.8M dataset, then used to fine-tune Mistral-7B as [[Orca 3]].


----

# Arena Learning / Wizard Arena 🪄🧙
- In the [[WizardArena|Arena Learning]]/[[WizardArena]] paper, we take a different approach to generating high-quality instruction data.
	- Instead of creating instructions from scratch, it focuses on refining existing instruction datasets through a simulated competitive environment.
	- It's not an agentic framework because tools aren't provided to the models, but it could easily be transformed into one.
![[Pasted image 20240719005227.png]]

The key components of the Arena Learning pipeline are:
1. Offline Pairwise LLM Arena: Arena Learning creates a simulated arena where multiple LLMs compete against eachother on a large set of instruction data. A judge LLM (LLaMA-3-70b-Instruct) evaluates the responses from competing models for each instruction, providing rankings, scores, and explanations. This process (with use of the WizardArena test set) effectively simulates human evaluation but at a much larger scale and lower cost.
2. Data Collection and Preprocessing: The framework starts with a large corpus of conversational data collected from various open sources, which goes through filtering/cleaning/deduplication. The refined dataset is then split into multiple parts for iterative training.
3. Iterative Battle and Model Evolution: The process involves multiple rounds of battles and training
	1. An initial model is trained on some subset of data.
	2. This model competes against other SoTA LLMs on another subset of data.
	3. Instances where WizardLM-B loses are collected, with the the winning model response used as the target for finetuning (Both SFT and DPO/PPO).
	4. The process repeats for multiple iterations, with each iteration potentially using different training strategies (SFT, DPO, PPO).
		- ((I don't recall this fact))
4. Training Strategies: Arena Learning employs multiple training strategies to improve the model
	1. SFT: We finetune our model on the winning battler's generation
	2. DPO: Treat win/loss responses and choice/reject pairs for training.
	3. PPO: Use battle results to train both a reward model and the language model.
5. WizardArena Evaluation: Authors create an offline test set (WizardArena) with diverse and hard subsets. This is used to evaluate models through pairwise battles, with results used to compute Elo rankings. The evaluation closely aligns with human-based arenas, but is much faster and cheaper.
	- ((Annoyingly, I don't believe that Authors gave many hints as to how they created the WizardArena test set, which is highly correlated with [[ChatBot Arena]] scores))
6. Data Selection: The pipeline uses various strategies to select high-quality training data, like threshold-based filtering to control data size and quality, focusing on instances where the model underperforms, and gradually shifting towards more complex data in later iterations.

![[Pasted image 20240719010914.png|400]]

This framework allows for multiple iterations of battles and training ((though they don't do enough iterations to really show us if 3 iterations is when the performance tapers off or not))

==Arena Learning== focuses on improving agents where the model under training is currently lacking. A nice feature is that it doesn't require particularly powerful models... the entire pipeline can be deployed using open-weight models, which is a big advantage.


---

# 🤝ArenaInstruct: Combining AgentInstruct and Arena Learning
- While both AgentInstruct and ArenaLearning aim to generate high-quality data for post-training, they take fundamentally different approaches!

Data Generation: 
- [[AgentInstruct]] starts from raw text documents, generating instructions from scratch through a multistage pipeline, allowing for the creation of entirely new content, potentially leading to greater diversity and novelty in the generated instructions.
- [[WizardArena|Arena Learning]] refines existing *instruction datasets* through simulated battles between models. Leverages the quality of existing datasets while improving on them through competitive evaluation.
	- ((To be clear, it's just refining the response side of the dataset, not the prompt side!))

Data Quality
- AgentInstruct relies on suggester-editor agent pairs for iterative refinement of instructions, allowing for fine-grained control over the complexity and quality of generated instructions.
- ArenaLearning uses an LLM as a judge to evaluate responses in simulated battles. This means the entire data quality process is handled by a *single model*.

Diversity and Complexity
- AgentInstruct explicitly designs for diversity through a human-written taxonomy of instruction types, and multiple transformation agents. This structured approach covers a wide range of skills and instruction types.
- ArenaLearning's diversity comes from the variety of competing models and initial instruction datasets. This might lead to less structured diversity.

Flexibility: 
- AgentInstruct's pipeline allows for easy addition of new seed types and instruction categories, making it highly adaptable to new domains and tasks. 
- ArenaLearning's iterative battle process enables continuous improvement of the target model, potentially allowing it to adapt more quickly to new challenges and competing models.

A taxonomy-based data generation is more steerable, and could be improved upon by area learning... but we could also use feedback signals to improve this first step over multiple iterations.
![[Pasted image 20240719142317.png]]
1. AgentInstruct Instruction Generation: Use AgentInstruct to create a broad and diverse base of instructions (no answers!) from raw text.
2. Arena Learning Answer Generation: Apply Arena Learning's competitive battle approach to refine and select the highest quality answers from a pool of models.
3. Data Quality Evaluation: Instead of relying on a single LLM as a Judge, we can rely on [[Reward Model]]s, or use *LLM as as Jury* (An ensemble of diverse LLM models) to improve data selection.
4. ==Diversity Feedback==: ==Use insights from Arena Learning battles to dynamically update AgentInstruct's instruction taxonomy!== Focus the generation process on producing more of the instruction types that prove most challenging or useful in real-world scenarios.
5. ==Complexity Feedback==: Leverage Arena Learning's performance metrics to identify areas where instructions are too easy or too difficult. Use this information to guide AgentInstruct's complexity refinement process, ensuring a well-balanced dataset that challenges the model appropriately over several iterations.





