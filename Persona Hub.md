June 28, 2024
[[Tencent AI]] Lab, Seattle (*Chan et al.*)
[Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/abs/2406.20094)
#zotero 
Takeaway: Authors use a Text-to-Persona ("What persona would like/hate/laugh at/etc. this text) and a Persona-to-Persona (Given a persona and a relationship, generate another persona), plus some MinHash/Embedding deduplication to generate a collection of ~1B diverse "personas". You can then use these personas in any sort of data generation process to increase data diversity; authors do it for math, logical reasoning, tool use, and Quora article generation.


---
## Introduction
- In practice, it's easy to scale up the *quantity* of synthetic data, but it's difficult to ensure that its *diversity* scale up as well.
- Without considering sampling techniques, an LLM can only produce 1 instance given a data synthesis prompt. Therefore, to create diverse synthetic data at scale, a large number of diverse prompts are needed.
- Previous research tends to diversify data synthesis prompts through the following two paradigms (neither of which achieve scalable synthetic data creation):
	1. ==Instance-driven==: Diversifies the data synthesis prompt by leveraging some seed corpus (eg [[Self-Instruct]], [[Web Rephrase Augmented Pre-training|WRAP]]); the diversity of the synthesized data mainly comes from the seed instances, making it hard to truly extend *beyond* the seed corpus.
	2. ==Key-point-driven==: Diversifies the data synthesis prompt with a curated comprehensive list of key points (or concepts) that can be a topic, subject, or any knowledge we expect synthetic data to encompass (eg [[AgentInstruct]]). It's practically prohibitive even for a team of experts to curate a comprehensive list by enumerating all key points at different levels of granularity, though.
- Fortunately, a ==Persona== tactic is easy to scale up. Authors construct [[Persona Hub]], a collection of 1 billion diverse personas that they hope can tap into almost every perspective encapsulated in the LLM. 
	- ==Personas can be combined with almost any data synthesis prompt, benefitting from LLMs' strong roleplaying ability, making them a generally-applicable technique.==
	- Though the entire collection is 1 billion personas, authors initially release ==200,000 personas==  from Persona hub, and use them to create:
		- 50,000 math problems
		- 50,000 instructions
		- 10,000 game NPCs
		- 50,000 logical reasoning problems
		- 10,000 knowledge-rich texts
		- 5,000 tools (functions)

## Persona Hub
- ==Text-to-Persona==
	- A person with specific professional experiences and cultural backgrounds will have unique interests in reading and writing... so given a specific text, can can infer a specific persona who is likely to {read/write/like/dislike/...} it. Using this technique, we can generate a bunch of persona on web data.
	- There are many formats and granularities with which we could represent a persona.
		- Coarse-grained: "a computer scientist"
		- Fine-grained: "a machine learning researcher focused on neural network architectures and attention mechanisms"
		- We ask the LLM (in the prompt) to output as specific persona descriptions as possible (though we notice that input texts with many detailed elements also result in more specific personas).
- ==Persona-to-Persona==
	- The Text-to-Persona strategy might still miss some personas that have low visibility on the web, and thus are less likely to obtain (eg a child, a beggar, a behind-the-scenes crew member of a movie).
	- The Persona-to-Persona method derives additional personas with interpersonal relationships from those obtained through Text-to-Persona.
	- The persona of "a child" can be derived from the persona of a nurse at a children's hospital (patient-caregiver relationship). Similarly, a "beggar" can be derived from persona of a shelter worker (via an assistance relationship) and a "behind-the-scenes movie crew member" can be derived from the persona of a movie's lead actor (co-worker relationship).
	- ==According to the "six degrees of separation theory", we perform 6 iterations of persona relationship expansion for each persona obtained through TExt-to-Persona, enriching our persona collection even further.==
- Deduplication
	- We run ==text-to-persona== on the [[RedPajama v2]] dataset, then perform ==persona-to-persona== as described above. It's inevitable that these personas will be identical or extremely similar, so we deduplicate in two ways:
	1. [[MinHash]]-based Deduplication:
		- We deduplicate based on the n-gram features of persona descriptions. 
		- Since persona descriptions are usually just 1-2 sentences, much shorter than a document, we simply use ==1-gram and a signature size of 128 for MinHash deduplication, deduplicating at the similarity threshold of 0.9.==
	2. Embedding-based Deduplication:
		- After deduplication based on surface forms (i.e. MinHash with n-gram features), we also adopt embedding-based deduplication.
		- We use a text embedding model (OpenAI's `text-embedding-3-small`) to compute an embedding for each persona, and then ==filter out personas with a cosine semantic similarity greater than 0.9.==
			- ((This doesn't say which one you keep... a random one?))
		- If we needed more diversity and fewer samples, we could further apply a stricter deduplication standard (eg 0.5).
	- After this filtering, the authors are left with a harvest of 1,015,863,523 personas, forming [[Persona Hub]].

## Persona-driven Synthetic Data Creation
- Our proposed persona-driven data synthesis approach is straightforward and effective, which involves ==integrating a persona into the appropriate position in a data synthesis prompt==.
	- As simple as this appears, it can significantly influence the LLM to adopt the persona's perspective to create synthetic data.
- See Figure 6
	- ==Zero-shot prompting== does not leverage any existing examples (i.e., demonstrations), thereby fully exploiting the model’s creativity without being constrained by specific examples.
	- ==Few-shot prompting== can better ensure that the synthesized data meets the requirements by providing some demonstrations.
	- ==Persona-enhanced few-shot prompting== is more effective in enhancing the LLM’s persona-driven data synthesis capabilities. However, its drawback is that it requires deriving the corresponding persona for each demonstration in the few-shot prompt beforehand.

## Use Cases
- We demonstrate the use cases of Persona Hub in various data scenarios
	1. Math problems
	2. Logical reasoning problems
	3. Instructions (ie user prompts)
	4. Knowledge-rich texts
	5. Game NPCs
	6. Tool (function) development
- For math, they use 1.09M personas to create 1.09M math problems.
	- A 7B model finetuned on the synthetic training data achieved an impressive 64.9% on [[MATH]], outperformed only only by models like GPT-4o, Claude 3.5 Sonnet, DeepSeek-Coder-V2-Instruct.
	- Authors observe that the semantic similarity between synthesized math problems tend to be correlated with but lower than the similarity between their corresponding personas.
- Authors also release 50,000 synthetic reasoning problems.
- The end users of LLMs are ultimately humans; we can use Persona hub to simulate typical requests for LLM assistance.
	- "You're a helpful assistant. Guess a prompt that the following persona may ask you to do: {persona}"
	- There's also a Persona-Enhanced Few Shot version where the examples also come with personas, eg:
```
((An example of an exemplar))
Persona: A curious and analy;cal individual, likely with a background in mathema;cs or science, who enjoys exploring intriguing "what if" scenarios and is fascinated by the intersec;on of popula;on demographics and geography. Prompt: Is it possible for the global popula;on to stand on Jeju Island?
```
- Authors also use the Personas to generate knowledge-rich plain text benefitting pretraining/post-training of LLMs, like Quora or Wikipedia articles.
- Authors also use the Personas to generate "Tools", which are high-level interfaces (eg a json dict with function name, args, returns).
	- See the figure below, there seem to be some consistency problems with the JSON?


Abstract
> We propose a novel persona-driven data synthesis methodology that leverages various perspectives within a large language model (LLM) to create diverse synthetic data. To fully exploit this methodology at scale, we introduce ==Persona Hub== -- a ==collection of 1 billion diverse personas automatically curated from web data==. These ==1 billion personas (~13% of the world's total population), acting as distributed carriers of world knowledge, can tap into almost every perspective encapsulated within the LLM, thereby facilitating the creation of diverse synthetic data at scale for various scenarios==. By showcasing Persona Hub's use cases in synthesizing high-quality mathematical and logical reasoning problems, instructions (i.e., user prompts), knowledge-rich texts, game NPCs and tools (functions) at scale, we demonstrate persona-driven data synthesis is versatile, scalable, flexible, and easy to use, potentially driving a paradigm shift in synthetic data creation and applications in practice, which may have a profound impact on LLM research and development.


# Paper Figures

![[Pasted image 20240723185617.png]]
See that a given persona can be used to generate questions across a variety of use types (here, math program, logical reasoning, user prompt to LLM).

![[Pasted image 20240723192018.png|500]]
I mean... that captious is a little hokey, especially the last part. Okay man.

![[Pasted image 20240723193442.png|500]]
Describing the ==Text to Persona== task, in which, given a text corpus, we ask: "Who is likely to (read, write like, dislike) this text?" to generate some description of a persona.

![[Pasted image 20240723193806.png|500]]
Some examples of fine-grained persona descriptions that can result from the Text-to-Persona approach

![[Pasted image 20240723203913.png]]
The ==Persona-to-Persona== approach aims to capture some of the personas that aren't well-represented by text on the the internet... and it derives additional personas with interpersonal relationships from those obtained through the Text-to-Persona step.

![[Pasted image 20240723210003.png]]
- An example of ==zero-shot prompting example==, in which they simply provide the persona and ask it to generate a challenging (eg math) problem.
- A ==few-shot prompting example==, where we give two math problems, and then ask it to create another, but conditioned for a "chemical kinetics engineer"
- A ==persona-enhanced example==, in which we generate a persona for each of the demonstrations beforehand.

![[Pasted image 20240723211114.png|500]]
Using the same linguist persona to create a variety of math problems (prompting for a subtype of math problems in each)

![[Pasted image 20240723211322.png|500]]
Examples of math problems created using personas of professional *related to the field of mathematics!* Interestingly, the authors say they tend to be more challenging than those created with general personas.
- (Though we might like the diversity provided by all sorts of personas creating math problems)

![[Pasted image 20240723212135.png|600]]
MATH results when training on synthetic MATH data generated using 1.09M diverse personas (unsure how these were selected; are the random, or are they those closest in space to the "math professor" embedding, for instance?)

![[Pasted image 20240723214647.png]]
Asking to create a logical reasoning and spatial reasoning problem, conditioned on disparate personas (golfer, SWE). Interesting how granular these two personas are... Feels like it requires a mildly strong model to actually take full advantage of it, and still provide a reasonable response?
- It's interesting that the right example didn't really incorporate any aspect of "the social impact of their algorithms"


![[Pasted image 20240723215027.png|400]]
Example of user prompt generation, where we just ask what instruction a {persona} may ask a helpful assistant. There's also a persona-enhanced few shot version.

![[Pasted image 20240723220456.png|500]]
Using Personas to generate "knowledge-rich plain text" (eg Quora, Wikipedia) using Personas.

![[Pasted image 20240723220813.png]]
Using Personas to generate "tools" (high-level interfaces"... If you look, the input_args are all different? Like one is a dict, the other a string, and the other a list? Maybe they just needed to be more specific about the output/use Instructor.)
