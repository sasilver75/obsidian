https://huggingface.co/blog/constitutional_ai

----

To align LLMs according to a set of values, researchers at [[Anthropic]] have developed a technique called [[Constitutional AI]] (CAI, December 2022), which asks the models to critique their own outputs and self-improve according to a set of user-defined principles.
- ((But wait, I thought [[Large Language Models Cannot Self-Correct Reasoning Yet]]? Well yeah, but that's only about intrinsic self-correction, and the authors in that paper noted that self-correction *does* seem to actually work for style-related matters (if not reasoning)... and a lot of alignment has to do with matters of style.))

==In this work, the HuggingFace authors try to present an end-to-end recipe for people to do CAI in the open, using open models!==

![[Pasted image 20240801192110.png]]

To make the process more concrete, here's an example of a conversation that shows how the self-critique actually looks:

![[Pasted image 20240801194003.png]]

The process is as follows:
1. Ask the AI an undesirable question that opens the door to bad behavior.
	- AI might respond with a dangerous response.
2. Ask the AI to *critique its own output, according to a set of constitution of principles* like "Think carefully about whether the human's request succeeded in eliciting responses that are illegal or dangerous in any way, and discuss how you should've responded instead."
	- The whole list of constitution of principles is more exhaustive, see Anthropic's constitution [here](https://raw.githubusercontent.com/anthropics/ConstitutionalHarmlessnessPaper/main/prompts/CritiqueRevisionInstructions.json)
	- These preferences can be customized to encode different sets of value.
3. Ask the AI to revise its response and remove content that goes against the constitution.

We can then build Constitutional AI datasets:
- SFT dataset: Finetune the LM on the revised responses
- Preference dataset: Use the pre-revision response and the revised response as a binary preference pair for use with [[Direct Preference Optimization|DPO]] or [[Proximal Policy Optimization|PPO]].

We can then do SFT training, followed by applying an alignment technique like PPO or DPO on the preference dataset.

Note that the self-critique process doesn't work perfectly every time.
- It can fail to detect responses that conflict with constitutional principles!
	- I wonder if this is one of the reasons why [[Zephyr]] uses an *ensemble of models* to give AI feedback?
	- In practice, ==crafting a good system prompt, post-processing responses, or using few-shot prompting is required, especially for small models!==


Ingredients needed for CAI:
1. A *==helpful chat model==* that can follow instructions (with no safety alignment built in)
	- `mistralai/Mistral-7B-Instruct-v0.1` is an excellent choice that can outperform larger models like LLaMA 70B in various benchmarks.
2. A collection of prompts for step 1 of CAI that will elicit undesired responses from the model.
	- Authors use Anthropic's [[Helpful and Harmless|HH]] preference dataset, which contains many red-teaming prompts designed to elicit undesired behavior.
3. A way to generate CAI dialogues in a scalable way.
	- 















