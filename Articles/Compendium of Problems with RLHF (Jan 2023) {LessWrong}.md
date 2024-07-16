#article 
Link: https://www.lesswrong.com/posts/d6DvuCKH5bSoT62DB/compendium-of-problems-with-rlhf

-----------

[[Reinforcement Learning from Human Feedback|RLHF]] TLDR: ==We need a reward function, but we cannot hand-write it -- let's make the AI learn it!==

## Problem 0: RLHF is Confusing
- Human judgement and feedback is so brittle that even junior alignment researchers like the author thought that RLHF is a not-too-bad-solution to the outer alignment problem.
- RLHF confuses a lot of people and distracts people from the core issues. Let's become less confused.

![[Pasted image 20240207231106.png]]

- Without RLHF, approximating the reward function for a "good" backflip is *tedious, to the point of being almost impossible!*
	- WITH RLHF, it's possible to obtain a reward function that, when optimized by an RL policy, leads to beautiful backflips.
	- RLHF withstands more optimization pressure than supervised fine-tuning! By using RLHF, you can obtain reward functions that withstand more optimization pressure than traditionally hand-crafted reward functions -- if that's something we care about.


### Why is RLHF insufficient?
- Buck two main problems with RLHF to create an AGI:
	1. Oversight issues
	2. Potential for catastrophic outcomes

Davidad suggests that dividing the use of RLHF into two categories:
- Vanilla-RLHF-Plan
	- Refers to the narrow use of RLHF (as used in ChatGPT), and is not sufficient for alignment
- Broadly-Construed-RLHF
	- Refers to the more general use of RLHF as a building block that might be potentially useful for alignment. 

Let's talk about the problems with "Vanilla RLHF" that "Broadly Construed RLHF" should aim to address:

### Existing problems with RLHF because of (Currently) non-non-robust systems

1. Benign Failures: ChatGPT can fail; it's not clear if this problem will disappear as models grow larger. (Eg )
2. Mode Collapse: ==A drastic bias towards a particular completion and patterns. Mode collapse is *expected* when doing RLHF!==
3. ==You *need* regularization==: Your model can severely diverge from the model you would have gotten if you'd have gotten feedback in real-time from real humans. 
	- ==You need to use a KL divergence and choose the constant of regularization. Choosing this constant feels arbitrary==. ==Without this KL divergence, you get mode collapse==.


My opinion:
- The benign failures seem to be much more problematic than mode collapse and the need for regularization!
	- Currently, prompt injections seem to be able to bypass most security measures.


#### Incentive issues of the RL part of RLHF
- I would expect these problems to be more salient in the future, as it seems to be suggested by recent work done at Anthropic.

4. RL makes the system more goal-directed, less symmetric than the base model.
	- Paul Christiano has a story about a model that was overoptimized at AI; it was trained using a positive sentiment reward system, and ended up determining that wedding parties were the most positive subject. Whenever the model was given a prompt, it would generate text that described a wedding party. 
		- Fine-tuning a large model with RLHF shapes a model that steers the sequence in rewarding directions. The model has been shaped to maximize its reward by any means necessary, even if it means suddenly delivering an invitation to a wedding party.


5. Instrumental Convergence
- Larger RLHF models seem to exist harmful self-preservation preferences, and ==sycophancy==; insincere agreement with user's sensibilities.

6. Incentive Deception
- ==RLHF incentivizes promoting claims based on what the human finds most convincing and palatable, rather than on what's true. RLHF does whatever it has learned makes you hit the "approve" button, even if that means deceiving you.==

7. RL could make thoughts opaque
- GPT's next-token-prediction process roughly matches System 1 (intuition) processes, and isn't easily accessible... but GPTs can also exhibit more complicated behavior through chains of thought, which roughly match System 2 (aka human conscious thinking processes).
- Humans should be able to understand how even human-level GPTs (trained to do next-token prediction) complete complicated tasks by simply reading the chains of thought. GPTs trained with RLHF will bypass this supervision.

8. Capabilities externalities
- RL already increases the capabilities of base models, and RL could be applied to further increase capability by a lot, and could go much further than imitation. 

9. RLHF requires a lot of human feedback, and still exhibits failures.
- To align ChatGPT, I estimated that creating the dataset cost 1M of dollars, which was roughly the same price as the training of GPT3.
- 1M is still not sufficient to solve the problem of benign failures currently, and much more effort could be needed to solve those problems completely.

10. Human operators are fallible, breakable, manipulable.
- Human raters make systematic errors, regular, compactly describable, and predictable errors.
- Aligning an AI with RLHF requires going through unaligned data; this leads annotators to psychological problems.

11. You are using a proxy, not human feedback directly!
- The model is proxy trained on human feedback, which represents what humans probably want -- you then use the model to give reward to a policy! 
- This is less reliable than having an ACTUAL human give the feedback directly to the model!

12. How to scale human oversight?


Using RL(AI)F may offer a solution to all the points in this section: By starting with a set of established principles, AI can generate and revise a large number of prompts, selecting the best answers through a chain-of-thought process that adheres to these principles. Then, a reward model can be trained and the process can continue as in RLHF.


# Superficial Outer Alignment

13. Superficially aligned agents: Bad agents are still there beneath; the capability is still here.

14. The system is not aligned at all to the beginning of the training, and has to be pushed in dangerous directions to train it. For more powerful systems, the beginning of the training could be the most dangerous period!

15. RLHF isn't a specification, only a process. RLHF is just a fancy word for preference learning, leaving almost the whole process of what reward the AI actually gets as undefined.

15. If you have the weights of model, it's possible to misalign it by fine-tuning it again in another direction; eg fine-tuning it so that it mimics Hitler, or something.
	- This is very inexpensive to do with LoRAs, etc. 
	- If you believe in the orthogonality thesis, you could fine-tune your model towards any goal!

The ==strawberry problem==: , "How would you get an AI system to do some very modest [concrete action](https://arbital.com/p/task_goal/) requiring extremely high levels of intelligence, such as building two strawberries that are completely identical at the cellular level, without causing anything weird or disruptive to happen?"

The Strawberry problem: RLHF doesn't a priori solve the strawberry problem.

17. Pointer problem: Directing a capable AGI towards an objective of your choosing.
18. Corrigibility: Ensuring that the AGI is low-impact, conservative, shutdownable, and otherwise corrigible.

At the end of the day, the author isn't worried about corrigibility at the moment. 


# Unknown properties under generalization

19. Distributional leap
- RLHF requires some negative feedback in order to train it. For large scale tasks, this could maybe require killing someone to continue the gradient descent? (EG if the model is making a choice in training about whether to save or kill someone, I guess). You could make train it in simulation or do curriculum learning to slowly increase the real-world stakes.
- Unknown properties under generalization: Even at the limit of the amount of data and variety you can provide via RLHF, when the learned policy generalizes perfectly to all new situations you can throw at it, the result will almost certainly be misaligned because there are still *near infinite* of such policies, and they each behave differently on the infinite remaining types of situations taht you didn't manage to train it yet on.
- Large Distributional Shifts to dangerous domains: RLHF doesn't a priori generalize optimize for alignment that you did in safe conditions, across a big distributional shift to dangerous conditions.
- Sim to real is hard: RLHF won't enable you to perform a pivotal act. Under the current paradigm, you would need the model to execute multiple pivotal acts, then assess each one. 
- High intelligence is a large shift: Once the model becomes very intelligent and agentic because of RLHF, this is akin to a large shift of distribution.


20. Sharp left turn
- How to safely scale the model if performances go up? Capability generalizes further than alignment. Some values collapse when self-reflecting. The alignment problem requires we look into the details of generalization. This is where all the interesting stuff is."










