
# Blog Ideas
- Embeddings (Different techniques, e.g. Matryoshka, Applications)
- Tokenizers (BPE, SentencePiece, WordPiece, ...)
- Synthetic Data (Self-Instruct, Evol-Instruct, OSS-Instruct, Orca, Orca2, Genstruct, Rephrasing the Web etc.)
- Decoding Strategies (Greedy, TopK, TopP, MCTS-reward-model-guided, LaMBDA classifier decoding)

# Project Ideas
- Implement [[Medusa]]


# Research Ideas
- The idea of speculative decoding, where you have a smaller model running in parallel to the larger model -- is there a way of doing something like RAG in this manner? Where you're increasing the speed of rag by speculatively retrieving documents that might be needed in the next (eg) paragraph of text?
- Question@JaredKaplan re: CAI: "Why use a scalar for the reward as opposed to anything else?" -> "Interesting research questions; could imagine a bunch of functions applied to the reward; imagine punishing bad behavior more extremely than good behavior, or changing the way that you sample. We've mostly done the simplest thing, but there's interesting research to be done on variations."
- [[Stanford Human Preferences|SHP]] uses numbers-of-upvote heuristics on Reddit data to construct a synthetic preference dataset. Is there a way to use some other heuristic (eg some classifier ensemble) (maybe combined with some rewriting) to create a similar dataset that captures some of the false negatives using semantic understanding?
	- (Oh, is this already what AlpacaFarm, UltraFeedback, Kim et al. 2023, Xu et al., 2023) do?
- Mixture of depth says that we can use smaller networks (iirc) to predict "Easier" tokens. Matryoshka lets us have these variable-dimensional versions of vectors; is there some sort of way of doing something similar with model parameters (some MRL-similar loss of some sort) so that we can be infinitely (?) variable in the amount of compute that we put towards a token?
- I loved the WizardLM paper. They complicated instructions by complicating the prompts themselves, at a... structural level. What if we were to complicate the prompts by asking the LM to ask X question as if they were Y type of person? Sort of inspired by those Youtube videos of "Context X explained to a grade schooler, high schooler, college student, phd student, professor"
- In the Lamini paper, when they're talking about Topic-Guided Instruction Generation, they constrain the topics that they look at because they worry that the long-titled topics are for obscure topics that the instruction-generating model won't know about. Can we use retrieval-augmentation to allow instruction-generating models to generate data?
- I was talking with LDJ about synthetic data for pretraining, and we were lamenting the fact that pretraining on synthetic data requires that you do twice the number of forward passes of models... I was curious, is there a way of self-alignment on "synthetic" data such that you (eg) beam search out a bunch of sentences, and then use a reward model to pick the best one, and then 
- https://www.youtube.com/watch?v=607EcmU9mFs&list=PL-Fhd_vrvisMYs8A5j7sj8YW1wHhoJSmW&index=2 This CMU MM course has a TA lecture on how to come up with research ideas in MM
- https://x.com/jd_pressman/status/1806520905532625084 Using Rejection Sampling and Backtranslation methods to create synthetic datasets?
- https://youtu.be/B45s_qWYUt8?si=R5GLWXTiBAwV9ImR&t=345 Doing search over synthetically generated prompts to find prompts that (verifiably? By a judge?) break the model, and then using training to "fix" that break.
	- "Because of Synthetic data, we've seen a total modal collapse of the "personality" of many models, since everyone is basically training on frontier model outputs. For Command R+, people say the model feels different/special, and that's not any magic they did at Cohere -- they just didn't do what everyone else does, which is train on the model outputs of OpenAI. No 'unraveling them mysteries,' no 'diving into the complexities,' etc."
	- "Is [the modal collapse of LM personalities...] because they're eating eachother's poop?" "Yeah, it's some sort of human centipede effect... Everything collapsing into [GPT-'4's] personality."
	- "Synthetic data methods that find more useful synthetic data... that are compelling at search, to automatically discover weak points of models, and closet those gaps."
- MCTS for Synthetic Data Generation?
	- ![[Pasted image 20240702225605.png|300]]
- > "I'm still of the mind that there do not exist good multi-turn datasets or other resources of bootstrapping anything like L3 or Gemma (without copious abuse of GPT-4 class models)." - Teortaxes
- There are a bunch of papers for synthetic IFT and IFT evaluation that try to create taxonomies of skills -- See AgentInstruct and Skill-Mix, and look for others for information. It would be nice to compile that into a blog post or something.
- Reagarding Synthetic Data pipelines, check out JD Pressman's [RetroInistruct](https://github.com/JD-P/RetroInstruct?tab=readme-ov-file) and [MiniHF](https://github.com/JD-P/minihf?tab=readme-ov-file#sampling-branch-management-and-autoloom) projects
- The [[SmolLM]] note talks about how for Cosmopedia v2, they started with a predefined list of 34,000 topics using the ==BISAC book classification==; that could be an interesting resource to check out, re: taxonomies.
- The FineWeb paper showed the power of training (eg) educational-quality classifiers to filter large datasets; they filtered FineWeb to FineWeb-Edu, which resulted in dataset that outperformed the original FineWeb dataset on popular benchmarks! Showed the power of classifiers trained on synthetic data... is there a way to push this further, somehow?
	- Stack-Edu-Python applied the same idea of FineWeb-Edu to Code! They used LLaMA3 to annotate 500k python samples from The Stack and used them to train an educational classifier using the same recipe as the FineWeb-Edu classifier.
- The process (simple AST mutation) that they use to created a PRM dataset for Code in Let's Reward Step by Step" seems pretty weak. Can we make something better, using some model-based synthetic data techniques and some filtering?
- What about in an iterative prompt refinement synthetic data pipeline, introducing another prompt or document and asking the LM to incorporate/fuse prompts together? Or having sort of abstract prompts (are these more like templates?) that we then couch in some context document.
	- Note that we can think about iterative refinement of prompts (AgentInstruct) and iterative refinement of responses (Arena Learning) separately.
	- Is there an advantage to using multiple models/prompts? AgentInstruct uses a suggestor and editor agent.
- Use of techniques like [[Constitutional AI|CAI]] critique together with synthetic data generation
- https://x.com/Teknium1/status/1815105755613376792
- The [[Better and Faster Large Language Models via Multi-token Prediction]] paper (in the figures) posits the idea that there are "choice points" of difficult to predict tokens whose selection heavily influences the following tokens... Is there some way to use per-token-reasoning ideas like those in [[Quiet STaR]] to target only 

https://x.com/thesephist/status/1734966611814289756?s=20

- Automatic Commit Message Generation
- Use the streaming LLMs with attention sinks paper to summarize some sort of live data feed, like financial or news information.
- As per [this](https://youtu.be/TIqf4LMNCjU?si=lTLl_4ft-TmCpMNr) video, distillation is usually used to create a smaller model with a similar performance. Can you use distillation (by providing higher quality labels; either distributions or rationales) to train successively better versions of the same size model? What about slightly increasing capacity models? Do you need more capacity to benefit from the slightly better labels?
- Sort of like how yikes in the Latent Space paper club used OpenInterpreter to scrape a bunch of related papers into a Weaviate rag chatbot (Verba), and then used it as a sidecar to the presentation.... could you do the same thing for a powerpoint presentation, for instance? Extract topics, etc from the slides, find relevant documents ingest them, and people can answer questions they have about the presentation from a little sidecar Q&A bot while the presentation is still going, so they can have better understanding.

