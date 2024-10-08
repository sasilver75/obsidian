The goal is to find:
- The seminal papers that we should be citing (eg Attention is All You Need)
- Reasoning Papers
- Self-Verification Papers
- Reasoning Recovery Papers

Thoughts:
- What about looking for this reasoning correction under different prompting strategies? Is it more likely if you use chain of thought versus few-shot prompting or zero-shot prompting?
	- I'm guessing we just want to use vanilla zero-few-shot Chain of Thought prompting in our project, so as not to make things too complicated?

Question:
- When we cite a paper date, do we cite the year in which the v1 was put on Arxiv, or when it was later accepted and published in a venue? I think it's the latter but that can be 1+ years away!


----
Dates may not be correct (usually going by v1 on Arxiv, but sometimes on published journal date)
(Wasn't sure if I wanted to include Eltuehr's GPT-J or Msft's Megatron-Turing NLG model too)

## Seminal Papers
- Vaswani et al. (2017) Attention is All You Need ([[Transformer]])
- Devlin et al. (2018) BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding ([[BERT]])
- Raffel et al. (2019) Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer [[T5]]
- Brown et al. (2020) Language Models are Few-Shot Learners ([[GPT-3]])
	- GPT-3 showing the ability to do few-shot learning
- Rae et al. (2021) Scaling Language Models: Methods, Analysis, and insights from training Gopher ([[Gopher]])
	- Just another language model, unsure if we want this (Deepmind)
- Bommasani (2021) On the opportunities and risks of foundation models
	- (If language models are going to be used for important things, failures in reasoning are an important problem to "solve")
- Ouyang et al. (2022) Training Language Models to follow Instructions with Human Feedback ([[InstructGPT]])
	- Uses both IFT and RLHF
- OpenAI (2022) ChatGPT: Optimizing Language Models for Dialogue ([[ChatGPT]])
- Chowdhery et al. (2022) PaLM: Scaling Language Modeling with Pathways ([[PaLM]])
	- Just another language model, unsure if we want this (Google)
- OPT: Open Pre-trained Transformer Language Models ([[OPT]])
	- Just another language model, unsure if we want this (Meta)
- Chung et al. (2022) Scaling Instruction-Finetuned Language Models ([[FLAN]]/[[FLAN-T5]])
	- Exploration of effect of instruction-finetuning on language models
	- Could also include: Longpre et al. (2023) The Flan Collection: Designing Data and Methods for Effective Instruction Tuning

- BigScience Workshop (2022) BLOOM: A 176B-Parameter Open-Access Multilingual Language Model ([[BLOOM]])
	- Just another language model, unsure if we want this (HuggingFace)

## Reasoning Papers (eg CoT)
https://github.com/Timothyxxx/Chain-of-ThoughtsPapers

 - Reasoning (And I don't think we need to cover every X-of-Thought strategy under the sun, here)
	- Rajani et al. (2019) Explain yourself! Leveraging Language Models for commonsense reasoning
		- This seems to be very similar to the later CoT paper. Authors include [[Richard Socher]]
	- Wei et al. (2022) Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
	- Wang et al. (2022) Self-Consistency improves Chain of Thought reasoning in Language Models ([[Self-Consistency]])
- Other inference-time prompting strategies that we probably won't look closely at:
	-  Zhou et al. (2022) Least-to-Most Prompting ([[Least-to-Most Prompting]])
		- "CoT performs poorly on tasks requiring solving problems harder than exemplars shown in prompts; we use a technique that breaks the problem down into simpler subproblems, then solves them in sequence"
	- Chen et al. (2022) Program of Thoughts Prompting
		- Interesting in that uses language models to express the reasoning process as a program, and then separately executes that code, which seems like it could help with ameliorate certain types of errors. Can be combined with self-consistency.
	- Zelikman et al. (2022) STaR: Self-Taught Reasoner Bootstrapping Reasoning with Reasoning ([[Self-Taught Reasoner|STaR]])
	- Yao et al. (2022) ReAct: Synergizing Reasoning and Acting in Language Models ([[ReAct]])
		- This one is a more "agentic" reasoning strategy, interacting with an external knowledge source over multiple terms.
	- Mukherjee (2023) Orca: Progressive Learning from Complex Explanation Traces of GPT-4 ([[Orca]])
		- Similar to STaR in that both involve generating (intrinsically, extrinsically) explanation traces and finetuning on the reasoning traces.
	- Yao et al. (2023) Tree of Thoughts: Deliberate Problem Solving with Large Language Models ([[Tree of Thoughts]])
		- CoT can still fall short in in tasks that require strategic look-ahead... ToT generalizes over popular CoT approach to prompting LMs... IIRC it's basically beam search + CoT.
	- Besta et al. (2023) Graph of Thoughts: Solving Elaborate Problems with Large Language Models
		- Models information generated by an LLM as an arbitrary graph, where LM "thoughts" are vertices, and edges correspond to dependencies between these vertices... I don't know if this is really a legit strategy, I've definitely heard less about it than CoT/ToT, and this is a buzzy space.
	- Shinn et al. (2023) Reflexion: LanguageAgent with Verbal Reinforcement Learning
		- A more "agentic" reasoning strategy I think, sort of in a similar place to ReAct.
		- I think it's kind of BS (relies on stronger evaluator and self-reflection models, iirc, and multiple rounds of iteration/guessing), Subbarao had bad things to say about both this and ReAct.
	- Zelikman et al. (2024) Quiet-STaR: Language Models can Teach Themselves to Think Before Speaking ([[Quiet STaR]])
- Math/Science:
	- Lekwoycyz et al. (2022) Minerva: Solving Quantitative Reasoning Problems with Language Models (Google Research)
	- Azerbayev et al. (2023) Llemma: An Open Language Model for Mathematics (Eleuther folks)
	- Trinh et al. (2024) Solving olympiad geometry without human demonstrations ([[AlphaGeometry]])
- Code:
	- Chen et al. (2021) Evaluating Large Language Models trained on Code ([[Codex]] paper)
	- Li et al. (2022) Competition-Level Code Generation with Alphacode ([[AlphaCode]])
	- Leblond et al. (2023) AlphaCode 2 Technical Report ([[AlphaCode 2]])
- PRM Papers
	- Solving math word problems with process- and outcome-based feedback (DeepMind)
	- Let's Verify Step by Step (OpenAI)
	- Let's Reward Step by Step (NUS)
- Critic Papers
	- Shepherd
	- Prometheus
	- Prometheus 2

## Self-Correction Papers
- Bai et al., (2022) Constitutional AI: Harmlessness from AI Feedback
	- Self-critique (IIRC?) but using some exogenous information in the form of a set of principles
- Manakul et al.  (2023) SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs
	- Instead of an approach that needs access to output probability distributions or external databases, this is a simple sampling-based approach, leveraging the idea that if an LLM has knowledge of a given concept, sampled responses are likely to be similar and contain consistent facts, but for hallucinated facts, stochastically sampled responses are likely to diverge and contradict one another.
- Miao et al. (2023) SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning
	- Exploring whether LMs can recognize errors in their own CoT without resorting to external resources.
- Madaan et al. (2023) Self-Refine: Iterative Refinement with Self-Feedback ([[Self-Refine]])
	- Uses a single LLM as the generator, refiner, and feedback provider
- Gou et al. (2023) CRITIC: Large Language Models can Self-Correct with Tool-Interactive Critiquing
	- Framework to validate and progressively amend their own outputs in a manner similar to human interaction with tools... Given initial output, interact with appropriate tools, then revise output based on feedback obtained during this validation process. Tested on QA/Math/Code/Toxicity Reduction
- Huang et al. (2023) Large Language Models Cannot Self-Correct Reasoning Yet
	- A short survey over self-correction methods (IIRC self-prompting, external oracle, multi-agent debate, ?) and shows that they don't meaningfully improve performance beyond self-consistency, IIRC, on GSM8K/CommonSenseQA/HotpotQA



## Dataset/Benchmark Papers
- Math
	- Cobbe et al. (2021) Training Verifiers to solve Math Word Problems ([[GSM8K]])
	- Hendrycks et al. (2021) Measuring mathematical problem solving with the MATH dataset ([[MATH]])
- Code
	- Chen et al. (2021) Evaluating Large Language Models Trained on Code ([[HumanEval]])
	- 

BIG Bench?
CommonsenseQA?



