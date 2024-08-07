---
aliases:
  - Reason and Act
---
October 6, 2022
Princeton, [[Google Research]] (Brain Team)
[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
#zotero 
Takeaway: Explores the use of LLMs to generate both *reasoning traces* as well as *task-specific actions* in an interleaved manner, allowing for greater synergy between the two (reasoning traces help the model induce, track, and update action plans, while actions allow the model to gather additional information) (reason-to-act, and act-to-reason, respectively). They basically just train a model on human-written examples of tool use, where human annotators basically just write down their thoughts "Okay, so I should search for X", "It doesn't look like this had what I need, what if I search for Y?" "This means that I should do X", etc. as they solve the problem.

Note: Subbarao Kambhampati [doesn't have nice things to say](https://youtu.be/hGXhFa3gzBs?si=MdW5opDIrgSt0Wr-) about ReAct's ability to generalize.

----
## Introduction
- Humans are unique in our ability to seamlessly combine task-oriented actions with verbal reasoning (or inner-speech) and maintaining a working memory.
	- ==When cooking in the kitchen, between any two actions, we might reason in language to track progress, handle exceptions, adjust the plan according to the situation, or recognize when external information is needed==. "Now that everything is cut, I should heat up the pot of water," "I don't have salt, so let's use soy sauce instead," "How do I prepare dough?"
	- ==We might also act to support the reasoning and to answer questions==. (Opening a cookbook to read the recipe, opening the fridge, checking ingredients, looking up some information online)
	- The tight synergy between "acting" and "reasoning" allows humans to learn new tasks quickly and perform robust decision making or reasoning, even under previously-unseen circumstances or facing information uncertainties.
- Properly prompted LLMs have demonstrated emergent capabilities to carry out several steps of reasoning traces to derive answers ([[Chain of Thought|CoT]]). But this CoT reasoning is a static black box, in the sense that the model uses its own internal representations/parametric knowledge to generate thoughts, so it isn't grounded in the external world, which limits its ability to reason *reactively*, or update its knowledge. ==This can lead to hallucination and error propagation over the reasoning process.==
- Recent work has explored the use of pre-trained LMs for planning and interacting in interactive environments (WebGPT, CALM, etc in the 2020-2022 era), with a focus on predicting actions via language priors.
- We present [[ReAct]], a general paradigm to combine reasoning and acting with language models for solving diverse language reasoning and decision making task.
	- We prompt LLMs to generate both verbal reasoning traces and actions pertaining to a task in an *interleaved manner*, which allows the model to perform dynamic reasoning to create, maintain, and adjust high-level plans for acting (==reason to act==, while also interacting with external environments to incorporate additional information in to reasoning (==act to reason==).


## ReAct: Synergizing Reasoning and Acting
- Consider a general setup where an agent receives an observation $o_t$ from the environment and takes an action $a_t$ following some policy $\pi(a_t|c_t)$, where $c_t = (o_1,a_1, ..., o_{t-1}, a_{t-1}, o_t)$  is the context to the agent.
- Learning a policy is challenging when the mapping from c_t -> a_t is highly implicit, and requires extensive computation/reasoning over the trajectory context.
- In ReACt, we augment the agent's action space with the space of language; any action in the language space we refer to as a *thought* or a *reasoning trace*. These don't affect the external environment, thus leading to no observation feedback.
- A thought $\hat{a}_t$ aims to compose useful information by reasoning or acting over the current context $c_t$, and update the context $c_{t+1}$ to support future reasoning or acting. Thoughts might include:
	- Decomposing task goals to create action plans
	- Injecting commonsense knowledge relevant to task solving
	- Extracting important parts from observations
	- Track progress and transit action plans
	- Handle exceptions and adjust action plans
	- ...etc.
- As the language space $L$ is unlimited, learning in this augmented action space is difficult and requires strong language priors -- we mainly focus on a setup where we use a ==frozen== LLM [[PaLM]]-540B, is prompted with few-shot in-context examples to generate both domain-specific actions and free-form language thoughts for task solving. Each in-context example is a *human* trajectory of actions, thoughts, and environment observations to solve a task instance.
	- For tasks where reasoning is of primary importance, we alternate the generation of thoughts and actions so that the task-solving trajectory consists of multiple ==***thought-action-observation***== steps.
	- In contrast, for decision making thats that potentially involve a large number of actions, thoughts only need to appear sparsely in the most relevant positions.
- ReACt enjoys several unique features:
	1. ==Intuitive and easy to design==: Designing ReACt prompt is straightforward, since human annotators just type down their thoughts in language on top of their actions taken.
	2. ==General and flexible==: Due to flexible thought space and thought-action occurrence format, ReAct works for diverse tasks with distinct action spaces and reasoning needs (QA, fact verification, text games, web navigation)
	3. ==Performant and robust==: ReAct shows strong generalization to new task instances while learning solely from one to six in-context examples, outperforming baselines that only use reasoning *or* acting.
	4. ==Human-aligned and controllable==: ReAct promises an interpretable sequential decision making and reasoning process where humans can easily inspect reasoning and factual correctness.

## Knowledge-Intensive Reasoning Tasks
- We begin with knowledge-intensive reasoning tasks like multi-hop question answering and fact verification.
- We consider two datasets challenging knowledge retrieval and reasoning:
	- [[HotpotQA]]: Multihop question answering benchmark that requires reasoning over two or more Wikipedia passages.
	- FEVER (2018): A fact verification benchmark where each claim is annotated to SUPPORTS, REFUTES, or NOT ENOUGH INFO, based on if there exists a Wikipedia passage to verify the claim.
- We use both of these in a ==question-only== setup, where models receive the question/claim as input *without access to support paragraphs,* and have to rely on their internal knowledge or retrieve knowledge via interacting with an external environment.
- For the action space, they design a simple Wikipedia Web API with three types of actions to support interactive information retrieval:
	1. ==search(entity)==: Returns the first 5 sentences from the corresponding *entity* wikipedia page, if it exists, or else suggests top-5 similar entities from the Wikipedia search engine
	2. ==lookup(string)==: Returns the next sentence in the page containing *string*, simulating Ctrl+F 
	3. ==finish(answer)==: Finishes the current task with *answer*
- We note that this action space mostly can only retrieve a small part of a passage based on exact passage name, which is significantly weaker than SoTA lexical/neural retrievers; the purpose is to simulate how humans would interact with Wikipedia, and force models to retrieve via explicit reasoning in language.
- For HotpotQA and FEVER, they randomly select 6 and 3 cases from the training set to manually compose ReAct-format trajectories to use as few-shot exemplars.
	- ((This is interesting -- remember that we're using only a frozen language model (without finetuning), so these exemplars are all we have to "teach" the model to act appropriately. I wonder why they chose such a small number of exemplars!))
	- The examples include thoughts that decompose questions, extract information from Wikipedia observations, perform commonsense/arithmetic reasoning, guide search reformulation, and synthesize the final answer.
- The authors notice that the problem solving process generated by [[ReAct]] is more factual and grounded, whereas [[Chain of Thought|CoT]] is more accurate in formulating reasoning structure, but can suffer from hallucination.
	- They propose to ==incorporate the two (CoT-Self-Consistency and ReAct),== with the following switching rule heuristics: 
		- ReAct -> CoT-SC: When ReAct fails to return an answer within given steps, back off to CoT-SC.
		- CoT-SC -> ReAct: When the plurality answer among n CoT-SC occurs less than n/2 times (i.e. internal knowledge doesn't support the task confidently), back off to ReACt.
- Authors consider a bootstrapping approach similar to [[Self-Taught Reasoner|STaR]], using 3,000 trajectories with correct answers generated by ReACt to finetune smaller language models (PaLM-8/62B) to decode trajectories (all thoughts, actions, observations) conditioned on input questions/claims.
- React vs CoT (Table 2 figure)
	- Hallucination is a serious problem for [[Chain of Thought|CoT]]; it's its major failure mode.
	- While interleaving reasoning, action, and observation steps improves ReACt's groundedness and trustworthiness, such a structural constraint also reduces its flexibility is formulating reasoning steps (an ==error pattern specific to ReAct== is that the model *repetitively generates the previous thoughts and actions, and we categorize it as part of the "reasoning error", as the model fails to reason about what the next proper action to take is, and jump out of the loop.*)
	- For ReAct, successfully retrieving informative knowledge via search is critical! Non-informative search accounts for 23% of error cases, and it derails the model reasoning and gives it a hard time to recover and reformulate thoughts.
		- ((I think there are a number of papers about making LMs robust to poor retrievals, and knowing when to rely on parametric knowledge))


## Decision-Making 
- We also test ReAct on two language-based interactive decision-making tasks, ALFWorld and WebShop, both of which feature complex environments that require agents to act over long horizons with sparse rewards, warranting the need to act and explore efficiently.
	- ALFWorld: A synthetic text-based game design to align with the embodied ALFRED benchmark, where tasks can have more than 50 locations and take an expert policy more than 50 steps to solve. "examine a paper under a desklamp," "go to coffeetable 1, take paper 2, use desklamp 1"
	- WebShop: A noisy real-world language environment.... an online shopping website with 1.18M real-world products and 12k human instructions -- "I am looking for a nightstand with drawers. It should have a nickel finish, and be priced lower than $140."
- ReAct outperforms Act on both ALFWorld and Webshop.


## Related Work
- [[Chain of Thought|CoT]] reveals the ability of LLMs to formulate their own thinking procedure for solving complicated tasks, zero-shot CoT, and reasoning with self-consistency.
- [[Self-Taught Reasoner|STaR]] bootstraps the reasoning process by finetuning the model on correct rationales generated by the model itself.
- Techniques like Scratchpad finetune an LM on intermediate computation problems.... 
- For decision making [[WebGPT]] uses an LM to interact with web browsers, navigate through pages, and infer answers to complicated questions.
- SayCan and Inner Monologue use LLMs for robotic action planning and decision making.
- ...




Abstract
> While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. [[Chain of Thought]] prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, ==we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner==, allowing for greater ==synergy== between the two: ==reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information==. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines, as well as improved human interpretability and trustworthiness over methods without reasoning or acting components. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. On two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples. Project site with code: [this https URL](https://react-lm.github.io/)


# Paper Figures
![[Pasted image 20240717175051.png|600]]
Note that in ReAct it seems like "thoughts" and "observations" seem to be different things? Though in the bottom-right, they don't have any "thought" example? OH, I see that it's a "type" of Act, and it seems to perhaps be optional. Is it locked into a Think/Act/Observe system, or can it choose whatever operation it wants? 

![[Pasted image 20240717222930.png]]
It seems for some reason seems like CoT is better at the "reasoning error" thing? OR wait, it just seems higher in React because React isn't spending 56% of its errors on Hallucinations like CoT is. So you just have to correctly interpret this chart, which doesn't have absolute error numbers, just percentages within a success/failure category.

![[Pasted image 20240717225028.png]]

![[Pasted image 20240717230119.png|600]]

![[Pasted image 20240717230214.png]]

![[Pasted image 20240717230244.png]]


# Non-Paper Figures

![[Pasted image 20240731105247.png]]
https://youtu.be/hGXhFa3gzBs?si=PMUXYm6e8XwinTB_ : Generalization is non-existant; they 