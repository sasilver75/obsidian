December 19, 2022 (8 months after [[Super-NaturalInstructions]]) -- [[Meta AI Research]]
Paper: [Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor](https://arxiv.org/abs/2212.09689)

A ==synthetic dataset== (unlike the previous, crowd-source IFT datasets of [[Natural Instructions]] and [[Super-NaturalInstructions]]) of ==240k instruction-following examples== (instructions, inputs, outputs)

- An automatically collected instruction dataset of 240k examples where [[InstructGPT]] (text-davinci-002) is prompted with three [[Super-NaturalInstructions]] examples (consisting of an instruction, input, and possible output constraints) and asked to generate a new example.
- Covers a ==more diverse set of tasks than Super-Natural Instructions==; while many examples reflect classical NLP tasks, it also includes other interesting tasks like Recipe Correction, Poem Generation, and more.

Abstract
> ==Instruction tuning== enables pretrained language models to perform new tasks from inference-time natural language descriptions. These approaches ==rely on vast amounts of human supervision in the form of crowdsourced datasets or user interactions==. In this work, we introduce ==Unnatural Instructions==: a large ==dataset of creative and diverse instructions, collected with virtually no human labor==. We collect ==64,000 examples== by *prompting a language model with three seed examples of instructions and eliciting a fourth*. This set is then *expanded by prompting the model to rephrase each instruction*, creating a total of approximately 240,000 examples of instructions, inputs, and outputs. Experiments show that despite containing a fair amount of noise, training on Unnatural Instructions rivals the effectiveness of training on open-source manually-curated datasets, surpassing the performance of models such as T0++ and Tk-Instruct across various benchmarks. These results demonstrate the potential of model-generated data as a cost-effective alternative to crowdsourcing for dataset expansion and diversification.

![[Pasted image 20240424004237.png]]