From [[Maxime Labonne]]
https://huggingface.co/blog/mlabonne/agentic-datagen

----

With the consolidation of LLM architectures, the quality of training data has become the most important factor in creating SoTA models. This is true for both pre-training and post-training.

Two innovative approaches have recent emerged to address the challenge of generating high-quality instruction datasets for post-training LLMs:
1. [[AgentInstruct]]
2. [[WizardArena]]/Arena Learning

Both frameworks come from [[Microsoft Research]] and leverage multiple LLMs to create and refine samples.

# AgentInstruct: A Multi-Agent Approach
- [[AgentInstruct]] (Mitra et al., 2024) is meant to generate large-scale, diverse, high-quality, and challenging data.
- Uses a pipeline that transforms raw text into refined instructions through multiple stages of processing.
![[Pasted image 20240719000942.png]]
Four main steps:
- ==Seed Collection==: Assemble a diverse collection of raw seeds, such as textbook chapters, web articles, and code snippets. These seeds serve as the foundation for generating new instructions.
- ==Content Transformation==: One or more specialized agents modify each seed into some *intermediate representation that simplifies instruction generation*. These agents perform tasks like generating argument passages, debates, conversations, meeting transcripts, poems, satirical content, etc.
- ==Seed Instruction Generation==: Multiple agents take the transformed seed and generate diverse instructions based on a pre-defined taxonomy of instruction types.
	- So in the domain of reading comprehension, the taxonomy includes 43 question types, ranging from literal comprehension to critical analysis and inference.
- ==Instruction Refinement==: 










