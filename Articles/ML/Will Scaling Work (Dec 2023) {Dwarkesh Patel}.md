---
tags:
  - article
---

Link: https://www.dwarkeshpatel.com/p/will-scaling-work

Formulated as a discussion between a ==Skeptic== and a ==Believer== about whether scale is all we need, and scale is achievable.

Questions asked: 
1. Will we run out of data?
2. Has scaling actually even worked so far?
3. Do models understand the world?
4. Will models get insight-based learning?
5. Does primate evolution give evidence of scaling?

----

# Will we run out of data?
### Skeptic
- We're about to run out of high-quality language data next year!
	- [[Epoch AI]] says that we may exhaust the stock of high-quality **language** data by 2026, and low-quality language data by 2030-2050. It also estimates that we might run out of vision data by 2030 - 2060. It also estimates that we might run out of vision data by 2030 - 2060.
		- [Chinchilla's Wild Implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications) argued that ==training data would soon become a bottleneck for scaling large language models==.  "Data, not size, is the currently-active constraint on language-modeling performance. Current returns to additional data are immense, and current returns to model size are miniscule! In fact, most recent landmark models are *wastefully big!*"
- Even handwaving scaling curves seriously imply that we'll 'll need 5 orders of magnitude (100,000x) more data than we seem to have (assuming same data efficiency).
	- Multimodal data will give us more data, and we can recycle tokens on multiple epochs, and use [[Curriculum Learning]].
	- But even if we assume the most generous possible one-off improvements that these techniques are most likely to give, they don't give us the 500 OOM that we need.
- Some say that we'll get the [[Self-Play]]/[[Synthetic Data]] working somehow, but ==self-play has two very difficult challenges==.
	- Evaluation
		- Self-play worked with AlphaGo because the model could easily judge itself based on some concrete win condition ("Did I win this game of Go?"). 
		- In contrast, ==novel reasoning doesn't always have a concrete win condition== that solutions can be evaluated against! And they can't really be trusted to evaluate their own outputs via "Self-Correcting"
			- Paper: [Large Language Models Cannot Self-correct Reasoning Yet](https://arxiv.org/abs/2310.01798v1) 
	- Compute
		- All the math/code approaches tend to use various sorts of tree search, where you run an LLM at each node repeatedly. This is very expensive!
			- AlphaGo's compute budget was staggering for the relatively circumscribed task of winning at Go.
				- Imagine if instead of searching over Go moves, you were searching over the space of all possible human thought.
### Believer
- In the believer's onion, the sample over which LLMs are "inefficient" is mostly just irrelevant e-commerce junk! And we compound this disability by training them on predicting the next token - a loss function that's almost completely unrelated to the actual tasks we want intelligent agents to do in the economy.
- And despite all of this, we still produced GPT-4 by throwing just .03% of Microsoft's yearly revenues at some big scrape of the internet!
- Researchers are only now getting around to making self-play work with current-generation models -- so the fact is that we just YET don't have enough public evidence about synthetic data's efficacy.
- ==Self-play doesn't require models to be ***perfect*** at judging their own reasoning. They have to be ***better*** at EVALUATING reasoning than at DOING IT de-novo.==
	- Almost all researchers that the Believer talks to are quite confident they'll get self-play to work.
### Skeptic
- Constitutional AI, RLHF, and other RL/self-play setups are good at bringing out latent capabilities (or suppressing naughty ones)... but no one has demonstrated a method to actually *increase the model's underlying abilities* with RL!
- If some kind of self-play/synthetic data *doesn't work*, you're fucked! There's no way around the data bottleneck! And I'm not sure if we should trust the researchers who have a financial interest in scaling working.

----
# Has Scaling actually even worked, so far?

### Believer:
- Of course! Performance on benchmarks has scaled consistently for 8 orders of magnitude!
- In the GPT-4 technical report, they said that they were able to predict the performance of the final GPT-4 model from "models trained using the same methodology, but using at most 10,000x less compute than GPT-4!" We should assume that this trend will continue to larger amounts of compute.
![[Pasted image 20240121173826.png]]

### Skeptic:
- Those benchmarks aren't good enough of a proxy, and (with respect to loss), we don't care directly about performance on next-token prediction! ==We want to find out whether these scaling curves on next-token prediction actually correspond to true progress towards general intelligence!==

### Believer: 
- As you scale the models, the performance consistently and reliably improves on a broad range of task as measured by benchmarks like [[Massive Multi-task Language Understanding|MMLU]], [[BIG-Bench]], and [[HumanEval]].

![[Pasted image 20240121174546.png]]
![[Pasted image 20240121174552.png]]

![[Pasted image 20240121174558.png]]

### Skeptic:
- Yeah, the numbers on the benchmarks are improving, but have you actually tried LOOKING at a random sample of questions from [[Massive Multi-task Language Understanding|MMLU]] and [[BIG-Bench]]? They're almost *all* just Google Search first hit results. They're good tests of *memorization*, not *intelligence!*

Here's some questions:

> Q: According to Baier’s theory, the second step in assessing whether an action is morally permissible is to find out
> 
> A: whether the moral rule forbidding it is a genuine moral rule.
> 
> Q: Which of the following is always true of a spontaneous process?
> 
> A: The total entropy of the system plus surroundings increases.
> 
> Q: Who was president of the United States when Bill Clinton was born?
> 
> A: Harry Truman

Why is it impressive that a model trained on a bunch of internet text memorized a lot of facts? In what way does this indicate intelligence or creativity?

- Even on these contrived and orthogonal benchmarks, it seems like performance is plateauing! ==Google's new [[Gemeni Ultra]] is estimated to have almost 5x the compute of GPT-4, but has almost equivalent performance== at MMLU, BIG-Bench, and other standard benchmarks.
- Common benchmarks don't measure long-horizon task performance (can you do a job over the course of a month), where LLMs trained on next-token prediction have very few effective data points to learn from. I
	- Indeed, as seen on performance on [[SWE-bench]] (which measures if LLMs can autonomously complete pull requests, integrating complex information over long horizons), LLMs aren't yet good at this.
- ==It seems like we have two kinds of benchmarks==:
	   1. The ==ones that measure memorization==, recall, and interpolation ([[MMLU]], [[BIG-Bench]], [[HumanEval]])... These models already appear to match or even beat the average human,
		   - As a result, these test CAN'T be a good proxy of intelligence, since these models are *clearly* much dumber than humans, right?
	   2. The ==ones that *truly* measure the ability to autonomously *solve problems* along long time horizons== or difficult abstractions ([[SWE-bench]], [[Abstraction and Reasoning Corpus]])
		   - For these, our models perform terribly.

In the skeptic's opinion, it's not even work asking yet whether scaling will continue to work, since in the skeptic's opinion, we don't have evidence tahat scaling has "worked" thus far (with respect to general intelligence, not memorization)

### Believer
- If there were some fundamental hard ceiling on deep learning and LLMs, shouldn't we have seen it before they started developing common sense, early reasoning, and the ability to think across abstractions? Why do you think there's some stubborn limit?
- Think about how much better GPT-4 is than GPT-3; that's just a 100x scaleup; which sounds like a lot, until you consider that ==we can totally afford a further 10,000x scaleup of GPT-4 (eg GPT-6-level) before we touch even *one percent* of world GDP==...
	- And that's before we account for 
		- pre-training compute efficiency gains ([[Mixture of Experts]], [[Flash Attention]])
		- new post-training methods ([[Reinforcement Learning from Human Feedback|RLHF]], fine-tuning on [[Chain of Thought]], [[Self-Play]])
		- hardware improvements

Adding all of these improvements together, we can probably convert 1% of GDP into a GPT-8-level model!

For context on *how much* societies are willing to spend on new general-purpose technologies:
- British railway investment at its peak in 1847 was a staggering 7% of GDP.
- Five years after the [[Telecommunications Act of 1996]] went into effect, telecommunication companies invested more than $500B (~$1T, today) into laying fiber optic cable and building networks.

You know the story -- millions of GPT-8 copies coding up kernel improvements, finding better hyperparameters, giving themselves boat-loads of high-quality feedback for fine-tuning, and so on.

# Do Models understand the world?

### Believer
- In order to predict the next token well, an LLM has to teach itself all the regularities about the world which lead to one token following another. 
	- To predict the next paragraph in a passage from *The Selfish Gene* requires understanding the gene-centered view of evolution; to predict the next passage in a new short story requires understanding the psychology of human characters.
- Gradient descent tries to find the most efficient compression of its data; the most efficient compression is also the deepest and most powerful. 

### Skeptic
- Intelligence involves (among other things) the ability to compress, but the compression itself isn't intelligence -- the *ability* to make compressions is. So if stochastic gradient descent makes compressions that become LLMs, that doesn't tell us much about the LLM's ability to make compressions (ie it's intelligence)

### Believer
- An airtight theoretical explanation for why scaling must keep working isn't necessarily for scaling to keep working. ==We didn't fully understand thermodynamics until a century after the steam engine was invented.== 
	- The ==*usual pattern* is that invention precedes theory==, and we could expect the same of intelligence.
- There's not some law of physics that says that Moore's Law must continue, but... it seems to!
![[Pasted image 20240121192631.png]]
- You can do all these mental gymnastics about compute and data bottlenecks, and the true nature of intelligence, or the britleness of benchmarks, or you can just look at the fucking line.


# Conclusion
- Enough with the alter egos, here's Dwarkesh's personal take!
	- If you were a scale believer over the last few years, the progress we've been seeing would have made more sense.
	- It seems pretty clear that *some amount* of scaling can get us to transformative AI -- i.e. if you achieve the irreducible loss on these scaling curves, you've made an AI that's smart enough to automate most cognitive labor (including the labor required to make smarter AIs)

Dwarkesh's tentative probabilities:
- 70% chance that scaling + algorithmic progress + hardware advances will get us to AGI by 2040.
- 30% chance that the skeptic is right and that LLMs and roughly anything in that vein are fucked.

==Observation:== The AI labs are simply not releasing much research, because any insights about the "science of AI" would leak ideas relevant to building the AGI. A friend who's a researcher at one of the labs said that he misses his undergrad habit of winding down with a bunch of papers -- nowadays, nothing worth reading is published!
- For this reason, Dwarkesh thinks that the things that he doesn't know would shortn his timelines.




# Appendix

### Will models get insight-based learning?
- At larger scale, models might naturally develop more efficient meta-learning methods; *grokking* might only happen when you have a large, overparamterized model, and beyond the point at which you've trained it to be severely overfit on the data.
	- Grokking seems similar on how we learn! We have intuitions and mental models of how to categorize new information
	- Over time, with new observations, those mental models themselves change.
- Skeptic:
	- NNs have grokking, but that's much less efficient than how humans actually integrate new explanatory insights. We're much more sample-efficient.

### Does primate evolution give evidence of scaling?
- Believer
	- Indeed, the human brain has as many neurons as you'd expect a scaled up primate brain with the mass of a human brain to have. Rodent and insectivore brains have much worse scaling laws; relatively bigger-brained species in those orders have far fewer neurons than you'd expect from just their brain mass.











