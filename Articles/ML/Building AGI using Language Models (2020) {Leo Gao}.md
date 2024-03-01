---
tags:
  - article
---


https://bmk.sh/2020/08/17/Building-AGI-Using-Language-Models/
Author: Leo Gao (https://twitter.com/nabla_theta): Cofounder and head of alignment memes @ ElelutherAI; currently RE @ OpenAI

----

Despite the buzz around GPT-3, it's not AGI. iN many ways, it's similar to AlphaGo or DeepBlue; it approaches human ability in one domain, but doesn't really seem like it will do "SCARY AI THINGS" any more than AlphaGo was going to turn the Earth into paperclips.

While its writing are impressive at emulating humans, GPT-3 doesn't have any memory of past interactions (on its own), and isn't able to follow goals or maximize utilities.
- But language modeling has a *crucial* difference from Chess or Go -- Natural language essentially encodes information about the entire world. By harnessing the world model embedded in the language model, it might be possible to build a *proto-AGI*.
	- Images aren't nearly as good as text is for encoding unambiguous, complex ideas, unless you put text *in* images (but then it's just language modeling with extra steps).
	- It seems likely that a sufficiently large *image* model could learn about the world through images, but likely at multiple orders of magnitude higher cost than an equivalent world-modeling-capability language model.

### World Modeling
-  The explicit goal of a language model is only to maximize likelihood of the model on natural language data.
- In the autoregressive formulation that GPT-3 uses, this means being able to predict the next word as well as possible.
	- This objective places much more weight on large, text-scale differences like *grammar* and *spelling* than on fine, subtle differences in semantic meaning and logical coherency, which reflect as very subtle shifts in distribution.
- At the extreme, any model whose loss reaches the Shannon entropy of natural language (the theoretical lowest loss a model could possibly achieve, due to the inherent randomness in language) will be *completely indistinguishable* from the writings of a real human in every way.
- The thing about GPT-3 that was so exciting is that it provided some indication that as long as we kept increasing model size, we can keep driving down the loss -- possibly even to the Shannon entropy of text -- without any new clever architectures or complex, handcrafted heuristics! By just scaling it up, we could get a better language model, and a better language model entails a better world model!
- But wait! Various experiments have shown that GPT-3 often *fails* that world modeling, and it doesn't seem obvious that adding more parameters will fix the problem! That's a valid critique -- a big assumption is the one that says that larger models will develop better world models!

### Putting the Pieces Together
- A world model doesn't alone make an agent, though! So what does it take to turn a world model into an agent! 
- It might be useful to set a goal, just as "*maximize the number of paper clips I have!*"
- Our model could do something like "I go to ebay, look up paperclips, sorted by price ascending. I spent $10 on the first item on the list." -- So it can probably generate some possible actions.
	- To estimate the state-action value of any action, we can do a [[Monte Carlo Tree Search]] (MCTS) to estimate the state-action values!
		- Starting from a given agent state, we roll out a sequence of actions using the world model. We integrate over all rollouts, and we know how much future expected reward the agent can expect to get for each action it considers. Then, we can just use (eg) a greedy policy with that state-action value function to decide on actions to take (with a greedy policy, this is just to always take the one with the highest state-action value.)

![Monte Carlo Tree Search visualized (<a href='https://www.researchgate.net/figure/Phases-of-the-Monte-Carlo-tree-search-algorithm-A-search-tree-rooted-at-the-current_fig1_312172859'>Source</a>)](https://bmk.sh/images/agi-lms/mcts.png)
*MCTS visualized*
- But each of these actions is likely to be pretty high level, such as "figure out the cheapest way to buy paperclips..." but thanks to the flexibility of language, we can describe very complex ideas with short sequences of tokens! 
- To actually execute these abstract actions once the agent decides on an action, that action can be broken down *further* using the language model into smaller *sub-goals* like "figure out the cheapest paperclips on Amazon dot com!"
	- Possibly even just directly breaking actions down into a detailed list of instructions would be feasible, depending on the capabilities of the model and how abstract the actions are.
		- {In other words, whether you have to recursive break an action down into a series of sub-actions or whether you can just go from actions to a list of instructions depends on your model capabilities, probably.}

- We could even represent the agent's state as natural language! Since the agent state is just a compressed representation of the observations, we can ask the language model to summarize the important information of any observations for its own internal world state {which... could be represented in text?}. 
	- The language model could even be used to periodically prune (ie forget) the information insides its state too, to make room for more observations!

- This would give us a system where you can pass observations from the outside world in, spend some time thinking about what to do, and output an action in natural language!
	- To handle input, you could have an input model that turns various modalities of observations into summarized text with respect to the current agent state.
		- How you do this is tangential to the point; what matters is that *somehow* the inputs are all converted to text and added to the agent state.
- To get the model to actually act in the world, you could again use the/a language model to translate natural language into *code that's executed*, or *shell commands*, or *series of keypresses*, or *anything!*
	- Like input, there are an infinitude of different ways to solve the output problem -- all that matters is that it's possible to get various modalities both *into* and *out of * the text-only agent.
		- {It sounds like text really *is* the universal interface 😉}

### Conclusion
- This has been more of a thought experiment than something that's actually going to happen tomorrow; GPT-3 today just isn't good enough at world modeling.
- 















