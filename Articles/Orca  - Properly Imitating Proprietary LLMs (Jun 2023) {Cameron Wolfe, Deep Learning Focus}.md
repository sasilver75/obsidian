#article 
Link: https://cameronrwolfe.substack.com/p/orca-properly-imitating-proprietary?utm_source=%2Fsearch%2Forca&utm_medium=reader2

-------

![[Pasted image 20240219211001.png]]

As research progresses on LLMs, one key question remains unanswered -- whether an existing, high-quality LLM can be used to effectively train another LLM.

The explosion of open-source imitation models post-LLaMA initially seemed to indicate that proprietary LLMs like ChatGPT could be easily replicated at low cost -- but subsequent research concluded that the evaluation of such models was incomplete and misleading, finding that these models actually have large gaps in their comprehension... 
- Essentially, models that used imitation learning (supervised training/fine-tuning off of responses from larger models) resulted in models that *seemed* to produce similar outputs to proprietary models, but they were actually more effectively copying the *style/structure* of responses than learning the *knowledge* that powered them.
- It was thought that there were two ways forward, for the open-source community:
	1. More capable base models to train from -- maybe if we were learning from extraordinarily capable base models like GPT-7, our imitation-learned models would be more capable!
		- It's been shown that more powerful models can benefit more from imitation learning than smaller models can.
	2. Create better datasets to learn from, effectively find some way to more effectively from the same base models.

The initial [[Orca]] paper is an attempt at #2, above.
- We will see that ==imitation learning can be made more effective by curating a larger dataset with more detailed information==.
	- ((Where ==imitation learning== is... just [[Supervised Fine-Tuning]] off of the responses from a more capable model))


# Background Information

## Instruction Tuning
- [[Instruction-Tuning]] was originally proposed by [[FLAN]], and aimed to provide a form of training that teaches LLMs to solve language-based tasks *IN GENERAL*, rather than any specific task.
	- This is done by fine-tuning an LLM over a set of "instructions," or input prompts, including a description of the task being solved, combined with the desired model output; see below.
	- ![[Pasted image 20240220000124.png]]
	- Above: ((So it seems that it's .... like task-specific fine-tuning, but you do task-specific finetuning over a variety of tasks? And then it still does pretty good versus out-of-distribution tasks))

Recent LLMs mostly use a specific variant of instruction-tuning that fine-tunes the LLM over examples of dialogue sessions -- either from humans or another LLM.
- This instruction-tuning usually occurs after pre-training.


Humans *can* manually create data for instruction-tuning, but ==we can also synthetically generate this instruction-tuning data using an LLM==. There are two basic approaches for this:
1. Obtain example dialogue sessions from another model 
	- (eg using ShareGPT, a site where people share interesting/good interactions with models like GPT-4)
2. Use a prompting framework (eg [[Self-Instruct]]) to *generate* and *refine* high-quality dialogue examples with an LLM.

Each of the above approaches are valid, but they each have limitatoins
- Public examples of LLM dialogues shared on the internet tend to be biased towards certain tasks, like creative content generation and information-seeking dialogue.
- Dialogues that were generated via self-instruct tend to lack complexity
	- But check out the [[Evol-Instruct]] strategy, which explicitly instructs and guides the LLM towards more complex generations -- See:
	- ![[Pasted image 20240220031018.png]]
	- Above: EvolInstruct


# The System Message

![[Pasted image 20240220031039.png]]
- Most chat-based LLMs that we interact with allow us to provide a system message ((==system prompt==)); see above. This message is basically an instruction to the model that describes how the model is expected to behave or respond to the user.
	- In the ChatGPT/GPT-4 APIs, this is given the "system" role, as opposed to either the "user" or "assistant."


Modern LLMs are steerable
- Earlier LLMs (including early versions of GPT-3.5-turbo) didn't pay much attention to the system message, but current models are much more ==steerable==, meaning we can provide detailed instructions in the system message for it to follow (talk like a pirate, return only JSON, etc.)
	- ==Steerability is a very helpful property when we want to use LLMs to generate specific synthetic data!==

# Other Useful Ideas
- ==Knowledge distillation and model imitation==
	- [[Distillation|Knowledge Distillation]] involves training a model from a more capable model's outputs.
- ==Packing technique==
	- (To handle variable-length sequences on hardware accelerators, it's common practice to use padding tokens so that all sequences in a batch have the same length. This results in up to 50% of all tokens being padding, which feels very inefficient. There are a variety of packing techniques to get around this fact. [link](https://arxiv.org/abs/2107.02027))
- ==Chain of Thought Prompting==
	- [[Chain of Thought]] prompting involves asking the LLM to produce a problem-solving rationale *before* providing the answer to a problem. Since LLMs build up the output content in an autoregressive fashion, chain-of-thought prompting can help them guide themselves to a correct answer.
	- Explanation tuning (more later on this) has lots of fundamental similarities to this technique.
- ==Curriculum learning==
	- Instead of just training a data over all the data we have, in [[Curriculum Learning]], we develop a particular strategy or curriculum for exposing this data to our model!
	- This may for example involve *first* training the model over dialogue examples from ChatGPT, then performing further training over GPT-4 dialogue. This term is quite generic, and many different types of curriculums exist.


# The Explosion 💥 of Open-Soruce LLMs 🦙

As LLMs began to increase in popularity, the most impactful models were initially proprietary models that were only available via paid APIs. There has since been a burgeoning open-source movement in the LLM community to create powerful open-source modells!


#### LLaMA and Imitation Models
- The weights of LLMs in the suite were openly released and then subsequently leaked online for anyone to access (via a troll pull request)
- An explosion of LLaMA + fine-tuning derivative models arrived:
	- Alpaca, Vicuna, Koala, GPT4ALL, ...
- These models were primarily created using an [[Imitation Learning]] approach, where we instruction-tune our model over dialogue examples from a more powerful model (eg ChatGPT). These models seemed impressive, and led the LLM community to believe that proprietary LLMs could perhaps be easily replicated -- however it turned out that even our best imitation models were primarily learning ChatGPT's *style*, rather than its *knowledge*, which oftentimes *tricked* the evaluators into thinking that they were talking to a smarter model than they actually were.
![[Pasted image 20240220032829.png]]
Above:  Because modern LLMs are *so good* at generating convincing text, the differences between them can be difficult to measure, especially when the models being compared have a similar style and fluency.

# Why doesn't imitation work like we want it to?
- The problem is generally: *We don't have enough high-quality data for instruction tuning.*
- To ameliorate this, we could:
	1. Ask smart humans to generate the data
	2. Use a prompting framework (eg Self-Instruct) to generate synthetic data
	3. Directly train on the outputs of existing LLMs

- Popular LLMs like GPT-4 are trained over extensive amounts of human feedback, but generating actual human feedback data can be expensive and time-consuming.
	- Recent imitation models have thus relied on synthetic fine-tuning dataset generation (eg using some variant of [[Self-Instruct]]).
	- Unfortunately, datasets generated in this manner tend to lack in diversity and complexity, and we run into similar problems when directly fine-tuning LLM dialogues obtained via public APIs, or places like ShareGPT.
### The Path Forward:
We basically have two options:
1. ==Create better open-source base models to serve as a better "starting point" for our instruction tuning.==
	- Some work has shown that using better underlying base LLMs drastically improves the performance of resulting imitation models.
	- ![[Pasted image 20240220033315.png]]
2. ==Consider ways to improve or expand existing datasets used for imitation learning. Current work relies solely on prompt and response pairs generated from an LLM... these are "shallow" imitations, only containing information about the LLM's answer/response to a prompt -- we can go beyond this by augmenting synthetic instruction tuning datasets with more detailed outputs, such as:== 
	- Explanation traces
	- Step-by-step thought processes
	- Complex instructions
- The imitation model can learn from this extra information produced by a proprietary model during fine-tuning.


# Properly learning to Imitate with Orca 🐋
- [[Orca]] is a 13B parameter model based on the LLaMA suite of LLMs, but is fine-tuned using more than just a small set of "shallow" imitation examples.
- More specifically, Orca differentiates itself in two main ways:
	- A ==larger and much more comprehensive imitation dataset==
	- ==Injecting detailed explanation traces into each instruction-tuning example.==
- ((TLDR: More, more-detailed instructions to learn from))
- The resulting model performs well across a variety of benchmarks, narrowing the gap between imitation models and proprietary LLMs.

Bigger and Better Data
- Orca selectively samples tasks from the [[FLAN]] collection, a massive data source for instruction tuning, and acquires millions of responses from both ChatGPT and GPT-4 over complex prompts from each of these tasks. When acquiring these responses, the authors use the system message/prompt to encourage the models to explain their response with added details, thus providing an =="explanation trace"== for each output generated by an LLM.
- Such an approach has a massive impact on model quality, and provides a richer source of information from which the imitation model can learn -- we refer to this approach as "==explanation tuning==."

# A better approach for imitation learning
- Orca's breakthrough in model quality can be attributed to its ==larger, more detailed, and more complex imitation dataset==. They did this by prompting more powerful models to output ==step-by-step problem solving explanations that are a much more powerful learning signal for imitating models== 
- Explanation Tuning
	- Prior imitation models are trained over pairs of prompts and associated responses (eg pointed "Yes", "No") generated from an LLM. These are =="shallow"== learning signals, and lack information regarding how or why a response was produced.
![[Pasted image 20240220142217.png]]
Above: Examples of prompts used in Orca to generate more detailed answers.

- To generate these answers, we prompt the teacher model to output detailed explanations prior to their normal responses -- These draw upon ideas like [[Chain of Thought]] prompting.
- We then use this information to fine-tune the imitation model using both the response and the explanations as a training signal.

Progressive Learning
- The dataset included 5M examples from ChatGPT (3.5) and 1M examples from GPT-4
- Instead of training on all of the data together, we achieved improved performance by fine-tuning Orca over the ChatGPT data first, and then fine-tuning on explanations from GPT-4 afterwards 
	- ((This is an example of human-intuition [[Curriculum Learning]], I think))
- Given that Orca's based on a smaller LLaMA model that's much less powerful than proprietary LLMs, this progressive learning approach allows the imitation model to first learn from "easier" examples, prior to learning from more "detailed" explanations of GPT-4.


# Explanation-based imitation learning is effective! 🧑‍🏫
- Orca is compared to a variety of different baselines, including [[Vicuna]], ChatGPT, and GPT-4
![[Pasted image 20240220142619.png]]
- We evaluate them on open-ended tasks, and Orca outperforms Vicuna by a large margin in all experimental settings.
![[Pasted image 20240220142636.png]]
- Above: See that Orca maintains 95% of ChatGPT quality, and 85% of GPT-4 quality across datasets. Although these metrics indicate significant improvement in performance compared to prior imitation models, we should keep in mind that LLM-based evaluations are imperfect ands till being explored.
	- ((Sam is curious about whether the questions/prompts that were used to explanation-tune this Orca model were at all similar to the questions that get asked on these evaluations -- if you, you trained an evaluation model, not a language model 😄))
	- ==Orca's performance still falls below that of ChatGPT in certain cases, but generally Orca performs similarly to ChatGPT across reasoning benchmarks. Across nearly all topics on the AGIEval and BigBench-Hard datasets, Orca comes near or exceeds the performance of ChatGPT!==


# Closing Remarks
- Research on open-source LLMs ins evolving -- it seems from week to week whether proprietary models are losing their moat or not. Every once and a while we remember the [[Bitter Lesson]].

Learning from step-by-step instructions
- Prior work on imitation models relied on "shallow" simple prompt-response pairs for training. ==We see here that instead augmenting this shallow data with detailed explanation traces allows the resulting model to learn better, from a much more rich source of information.==

Loss of imitation data
- One of the main problems with prior imitation models is that they only performed well on tasks that were similar to data seen in their fine-tuning datasets -- as a result, we needed larger imitation datasets with more coverage.
- Enriching our data slightly helped, but given this particular limitation, we clearly need larger imitation datasets with more coverage.

Remaining Work
- Orca's performance still falls short of the bets proprietary LLMs.





