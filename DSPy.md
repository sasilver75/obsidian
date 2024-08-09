October 5, 2023 (10 months after the OG Demonstrate Search Predict paper)
[[Omar Khattab]], Mattei Zaharia, [[Christopher Potts|Chris Potts]], et al. (Primarily Stanford)
Paper: [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)
#zotero 
Takeaway: DSPy is a new programming model for designing AI systems using pipelines of pretrained LMs and other tools to create text transformation graphs. Abstractions include signatures, modules, and teleprompters.

References:
- [Video: Vertex Venture's Neural Notes - Demonstrate Search Predict with Omar Khattab](https://www.youtube.com/watch?v=cih0eG_CmMY)
- [Video: Databrick's Data Brew - Demonstrate Search Predict Framework w Omar Khattab](https://youtu.be/bwkaI7olr_s?si=5zBBYNGPKJnzg6ox)
- [Video: Weaviate Meetup with Omar Khattab on DSPy](https://youtu.be/Y81DoFmt-2U?si=2zfYQqS0w3M3D6F8&t=2145)
- [Video: Designing Reliable AI Systems with DSPy | Zeta Alpha with Omar Khattab](https://youtu.be/Lba_MBZsR5s?si=hUH3LhKSQ6uj8rIM)
	
See also: 
- [[Demonstrate-Search-Predict - Composing retrieval and language models for knowledge-intensive NLP]] (Previous work, same authors)

---

Notes:
- There has been an explosion in interest around building multi-stage *pipelines* and *agents* that decompose complex tasks into more manageable calls to LMs in an effort to improve performance.
	- But LMs are sensitive in how they're prompted, and this is exacerbated in pipelines where multiple LM calls have to *interact* effectively. This results in brittle and unscalable hard-coded "prompt template" pipelines.
- In DSPy's programming model, we translate string-based prompting techniques (eg [[Chain of Thought|CoT]], [[ReAct]]) into declarative modules that carry natural language typed signatures.
	- DSPy modules are task-adaptive components that abstract any particular text transformation, like answering a question or summarizing a paper.
	- DSPy modules are ==parametrized==, so they can *learn* their desired behavior by iteratively bootstrapping useful demonstrations within the pipeline.
- The DSPy compiler optimizes any DSPy program to improve quality or cost. The compiler inputs are the program, with a few training inputs with optional labels, and a validation metric.
	- The compiler simulates version of the program on the inputs, and bootstraps example traces of each module for self-improvement, using them to construct effective few-shot prompts or finetuning small LMs for steps of the pipeline.
	- Optimization in DSPy is conducted by ==`teleprompters`==, which are general-purpose optimization strategies that determine how the modules should learn from data.
	- In this way, the compiler automatically maps the declarative modules to high-quality compositions of prompting, finetuning, reasoning, and augmentation.
- ==Overall, DSPy proposes the first programming model that translates prompting techniques into parametrized, declarative modules, and introduces an effective compiler with general optimization strategies (teleprompters) to optimize arbitrary pipelines of these modules.==
- Related Work: Authors note that researchers are starting to apply discrete optimization and RL to find effective prompts, generally for a single logical LM call; DSPy seeks to generalize this space, offering a rich framework for optimizing arbitrary pipelines using techniques from cross-validation to RL to LM feedback to Bayesian hyperparamter optimization.
- DSPY Programming Model
	- A DSPy program takes a task input (eg a question to answer, or a paper to summarize) and returns the output (eg an answer or a summary).
	- DSPy contributes three abstractions: 
		- `signatures`: abstract the input/output behavior of a module. 
		- `modules`: replace existing hand-prompting techniques and can be composed in arbitrary pipelines
		- `teleprompters`: optimize ll modules in the pipeline to maximize a metric.
- ==Signatures==
		-  A signature is a natural-language typed declaration of a function that says *what* a text transformation must do, rather than *how* ti should be one. A tuple of *input fields* and *output fields* (and an optional *instruction*). Akin to an interface/type signatures. We bootstrap useful demonstration examples for each signature, and handle structured formatting/parsing logic.
			- Can be expressed in a shorthand notion like `"question -> answer"`
		- The core module for working with signatures in DSPy is `Predict`; this stores the supplied signature, an optional LM to use, and a list of demonstrations for prompting. This instantiate module works like a callable function (like a layer in PyTorch), taking in keyword arguments corresponding to the signature input fields (eg `question`), formats the prompt to implement the signature, and includes the appropriate demonstrations, calls the LM, and parses the output fields.
		- There are a variety of more sophisticated modules like `ChainofThought`, `ProgramOfThought`, `MultiChainComparison`, and `ReAct`. These can all be used interchangeably to implement a DSPy signature.
			- All of these modules are implemented in a few lines of code by expanding the user-defined signature and calling Predict one or more times on new signatures as appropriate. So Chain of thought might look like: ![[Pasted image 20240517163647.png]]
	- DSPy parametrizes these prompting techniques! Any LLM call seeking to implement a particular signature needs to specify *parameters* that include:
		1. The specific LM to call
		2. The prompt instructions and the string prefix of each signature field
		3. Demonstrations used as few-shot prompts (for frozen LMs) or as training data (for finetuning).
		- By bootstrapping good demonstrations, we have a powerful way to teach sophisticated pipelines of LMs new behaviors systematically.
	- DSPy programs may use tools, which are modules that execute computation. We also support retrieval models through a `dspy.Retrieve` module, which supports [[ColBERTv2]] and others (probably many more by the time of my reading this)
	- DSPy modules can be composed in arbitrary pipelines: ![[Pasted image 20240517165031.png]]
	- We use ChainOfThought as a drop-in replacement of the basic `Predict`, above. Now we can simply write `RAG()("Where is Guarani spoken?")`
- ==Teleprompters==
	- When compiling a DSPy program, we generally invoke a *teleprompter*, which is an optimizer that takes a *Program*, *Training set*, and *Metric*, then returns a new optimized program.
	- Training sets may be as small as a handful of examples, though larger data enables better optimization. These training examples may even be *incomplete*, having only *input values*. We typically assume labels only for the program's final output, not the intermediate steps.
	- ==Metrics== can be simple notions like *exact match* (em) or F1, but they can also be entire DSPy programs that balance multiple concerns! 
	- The goal of optimization is to effectively bootstrap few-shot demonstrations: ![[Pasted image 20240517171256.png]]
	- Above: In this example, our BootStrapFewShot teleprompter simulates RAG on the training example(s) -- it will collect *demonstrations* of each module that collectively lead to valid output (i.e. respecting the *signature* and the *metric*).
	- If you wanted to push your compiled program to be *extractive* given its retrieved contents, we can define a custom metric to use in place of `dspy.evaluate.answer_exact_match` ![[Pasted image 20240517171616.png]]
	- Teleprompters can even be *composed* by specifying a `teacher` program! DSPy will sample demonstrations from this program for prompt optimization; This can enable very rich pipelines, where expensive program (eg complex expensive ensembles using large LMs) supervise cheap programs (eg simple pipelines using smaller LMs).
- ==DSPy Compiler==
	- A key source of DSPy's expressive power is its ability to compile (or automatically optimize) any program in this programming model. This relies on a teleprompter (an optimize for DSPy programs that improves the quality of modules via prompting or finetuning).
	- ==Typical teleprompters do through three stages:==
		1. Stage 1: Candidate Generation
			- The compiler first (recursively) finds all unique `Predict` methods (predictors) in a program, including those nested under other modules.
			- For each unique predictor *p*, the teleprompter may generate candidate values for the parameters of *p*: The instructions, field descriptions, and (most importantly) demonstrations (example input-output pairs).
			- In the simplest non-trivial teleprompter `BootstrapFewShot`, the teleprompter simulates a teacher program (or, if unset, the zero-shot version of the program being compiled) with some training inputs, possibly one or more times with high temperature. The program's metric is used to filter for results that help the pipeline pass the metric. We thus obtain potential labels for all signatures in the program by throwing away the bad example and using the good ones as potential demonstrations.
		2. Stage 2: Parameter Optimization
			- Now each parameter has a discrete set of candidates: demonstrations, instructions, etc.
			- Many hyperparameter tuning algorithms can be applied for selection among candidates.
			- Another type of optimization is *finetuning* with `BootstrapFineTune`, where the demonstrations are used to update the LM's weights for each predictor. Typically, we're optimizing average quality using the metric with cross-validation over the training set or validation set.
		3. Stage 3: Higher-Order Program Optimization
			- A different type optimization that DSPy compiler supports is modifying the control flow of the program! One of the simplest forms of these is ensembles, where we bootstrap multiple copies of the same program, and replace the program with a new one that runs them all in parallel and reduces their predictions into one with a custom function.

- Goals of evaluation
	- Our hypotheses
		- With DSPy, we can replace hand-crafted prompt strings with concise and well-defined modules, without reducing quality or expressive power.
		- Parametrizing the modules and treating prompting as an optimization problem makes DSPy better at adapting to different LMs, and it may outperform expert-written prompts.
		- The resulting modularity makes it possible to more thoroughly explore complex pipelines that have useful performance characteristics, or fit nuanced metrics.
- Conclusion
	- DSPy is a new programming model for designing AI systems using pipelines of pretrained LMs and other tools. Abstractions include signatures, modules, and teleprompters.




Abstract
> The ML community is rapidly exploring techniques for prompting language models (LMs) and for stacking them into pipelines that solve complex tasks. Unfortunately, ==existing LM pipelines are typically implemented using hard-coded "prompt templates",== i.e. lengthy strings discovered via trial and error. Toward a more systematic approach for developing and optimizing LM pipelines, we introduce ==DSPy==, a ==programming model that abstracts LM pipelines as text transformation graphs==, i.e. imperative computational graphs ==where LMs are invoked through declarative modules==. DSPy ==modules are parameterized, meaning they can learn (by creating and collecting demonstrations) how to apply compositions of prompting, finetuning, augmentation, and reasoning techniques==. We design a compiler that will optimize any DSPy pipeline to maximize a given metric. We conduct two case studies, showing that succinct DSPy programs can express and optimize sophisticated LM pipelines that reason about math word problems, tackle multi-hop retrieval, answer complex questions, and control agent loops. Within minutes of compiling, a few lines of DSPy allow GPT-3.5 and llama2-13b-chat to self-bootstrap pipelines that outperform standard few-shot prompting (generally by over 25% and 65%, respectively) and pipelines with expert-created demonstrations (by up to 5-46% and 16-40%, respectively). On top of that, DSPy programs compiled to open and relatively small LMs like 770M-parameter T5 and llama2-13b-chat are competitive with approaches that rely on expert-written prompt chains for proprietary GPT-3.5. DSPy is available at [this https URL](https://github.com/stanfordnlp/dspy)



---
[DSPy End-to-End Meetup](https://youtu.be/Y81DoFmt-2U?si=eCBDuT24Pyj83cmK&t=3124) Notes

![[Pasted image 20240523143552.png]]
The of optimizers is to maximize a particular function on your data.

![[Pasted image 20240523143901.png]]
Above: Many optimizers in DSPy
- MIPRO Optimizer: Multi-Instruction Proposal Optimizer
	- Bootstrap successful traces of the modules
	- Summarize patterns in data
	- Propose instructions based on these observations
	- Treat instructions x examples as hyperparameters across modules, applying a bayesian optimizer (TPE)


![[Pasted image 20240523144300.png]]
The program and optimizers can matter a lot more than your choice of language model. So when someone says "GPT-4 got X on Y benchmark," you really need to say "You don't know what you're talking about," if they're not talking about the program the model is running, and the optimizations that 
were used.

![[Pasted image 20240523145014.png]]
Note that he notes it being used for synthetic generation of retrieval training data - interesting. As well as for LLM-as-a-Judge-type applications.

![[Pasted image 20240523145227.png]]


![[Pasted image 20240523150043.png]]
Re: the 20-30 I/O examples... Don't collect chains of thought; no really weird examples. Just 20-30 I/O examples.
We can do a better job telling people how to build metrics. Maybe you *start* with something like "Hey GPT-4, help me judge this," and then get something better later on.
We have some guidance on which type of optimizer to select, but there are really only 3-4 that you should choose from at any given time. For optimizing, you might want to use someone else's data, or synthesize some data to match your task. You can figure out how to get some data for training


