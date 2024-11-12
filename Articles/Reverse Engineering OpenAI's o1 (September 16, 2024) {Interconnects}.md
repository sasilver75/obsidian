https://www.interconnects.ai/p/reverse-engineering-openai-o1

See: [[o1]]

---

In November 2023, the first leak occurred about a new model that was able to solve certain mathematical problems -- since then, OpenAI has been figuring out how to make this stable and package the results.

==Q*== was the original method for eliciting very high-value trajectories with some sort of tree-of-reasoning search. 

It's best to refer to o1 as a system -- there's a chance all operations are funneled through one advanced language model, but the funneling and cycling of those computations in a way that creates coherent outputs for the user is very complex.
o1 is still a preview
- OpenAI now has many angles they can take o1

> Our large-scale reinforcement learning algorithm teaches the model how to think productively using its chain of thought in a highly data-efficient training process. We have found that the performance of o1 consistently improves with more reinforcement learning (train-time compute) and with more time spent thinking (test-time compute). The constraints on scaling this approach differ substantially from those of LLM pretraining, and we are continuing to investigate them.

![[Pasted image 20241009102753.png|300]]

![[Pasted image 20241009102818.png|400]]
See that the released model (o1-preview) isn't as good as the yet-to-be-released o1 model.

Reasons for the o1-preview:
- OAI could not afford to serve the "strongest" configuration to the user.
- They do not have the infrastructure to deploy the final configuration.
- The final configuration isn't *safe* enough for their standards.
	- ((Though they seemed to show that o1 was actually *safer*, because of its reasoning abilities?))

![[Pasted image 20241009103031.png|400]]
From Jim Fan

This approach isn't good for every query -- eventually the ChatGPT product will absorb o1 and route your queries to the right model -- simple queries burn a lot of excess tokens in the system.

o1 is a useful demo to bridge the gap from existing language models to a world of different agentic products.
In the developer AMA an openAI rep said that o1 is a "model" and not a "system," but this might not tell the whole story.

When you read the few traces OpenAI provided, they're rambling and wandering forward to an answer.

Let’s formulate the reward for this RL problem. The classic problem with traditional reinforcement learning from human preferences is that one reward, in the form of a binary preference, is assigned to the whole trajectory. In this, it is hard to train the model to understand where it went wrong along the way. Recent research to get around this is by designing reward models that score every step in reasoning.

The best examples of per-step reward models ([[Process Reward Model]]) rating each step in a reasoning tree are from OpenAI’s _[Let’s Verify Step By Step](https://arxiv.org/abs/2305.20050)_ paper. The illustration below shows that the steps in green are high rewards and those in red are low rewards.

![[Pasted image 20241009103905.png|400]]

If you combine this with an outcome-based RM or heuristic that tells the system whether it got the answer right (and probably a length penalty, so it doesn't generate non-answers forever to avoid a negative return), the system will have a pre reasoning step reward assigning credit towards a final answer.

In the example above, reasoning steps could be regenerated from step 9, the first erroneous label, and then the RL agent has multiple similar trajectories to differentiate between based on a difference in reward.
Compared to traditional RLHF, full-state RL on language models will be more dependant on exploration and finding new reasoning steps. Without this, o1 could not continue to substantially improve its performance with more training time compute. More RL levels off, or even overfits and crashes, without sufficient exploration across a variety of states.

Nothing like this exists in the more open models. Reflection 70b style models look exactly like GPT-4o with special prompting. Not exploration. Most RL experts agree that ==exploration== is the most important aspect of an online learning agent.

If the model isn’t substantially larger, then we need to reason with the high inference costs for the model. o1-preview is charging ==$15 per million input tokens and $60 per million output tokens==. This is the same price as ==Claude 3 Opus== (at least at Opus’s launch, it may have come down). ==This also is applied to the intermediate reasoning tokens not shown to the user==. If the model isn’t bigger and not many tokens are generated, where is the compute going? My guess is a form of parallel decoding, which we can roughly call **==reinforcement learning language model decoding==** (to separate it from the usual autoregressive models).

==For each reasoning step shown to the user through the vague summary, o1 models generate multiple candidates that they then rate after an end-of-step token==. For users, this number is fixed. When doing evaluations, OpenAI can vary the number of candidates (and they said that want to expose this type of inference-intensity control to the user).

==This type of parallel and branching generation will take a notably different inference stack than normal chatbots, explaining high prices and “expectations for prices to drop==.” These form the actions mentioned above in the RL section and due to the flexible nature of the environment they can be widely varied in generation length.

![[Pasted image 20241009104834.png|500]]

==When you read the traces of this model it is very clear it’s different than any language model we’ve been playing with recently. It rambles, it questions, and it still gets to smart answers. It seems like there are some variable actions the model is taking, particularly at repetitive phrases like “Wait, is that right” or an oddly human “Hmmm.” These mark interesting moments when reasoning can change directions.==

> In the future we'd like to give users more control over how much time the model spends thinking.


What enables all of this on the research side is that reward models are being absorbed into generative language models. Recent research has shown [reward models that think out loud](https://arxiv.org/abs/2408.11791) before scoring or [very strong generative models for differentiating text](https://arxiv.org/abs/2408.15240). This type of implementation detail where all of Monte Carlo Tree Search can be handled by one final language model, makes things very confusing at inference time. A model generates potential ideas, scores them itself, and then generates a solution. At least at training time, it is almost a sure thing that more models are used as heuristics to rate the highest-quality reasoning traces.

==Creating an open copy of this system will be harder than it was for ChatGPT. Modular AI systems are sensitive to how they are connected together.==

==Additionally, the world is all looking. OpenAI knows this and is not showing the reasoning trace to users and sending cease and desist emails to users trying to jailbreak them out of the model. Getting the right seed data is the crucial part to building this.==
  
==Over a year ago, OpenAI likely paid high-skill annotators to create complex forward reasoning paths, likely with different paths for single problems. These can be rated and turned into initial labeled trajectories. It is likely that contrastive examples were needed too, making copying the reasoning traces (if we had them) not enough.==

At the end of the day, this is still training a language model, and _mostly_ fine-tuning one at that. When fine-tuning a model, you can tend to see behaviors emerge with just 1000s of samples. I expect this is the same for o1-like models==. Those 1000 will be extremely expensive, but it is a tractable goal.==

In the open, we’ll need to break down every piece of this system into a tractable problem. We need a state space formulation for wandering reasoning. ==We need good process reward models== for labeling trajectories. We need ==models that can generate a diversity of reasoning steps==. We need the ==compute to put this all together==. We need ==much harder evaluations to hill climb on==. It’ll take many small bricks to start building this picture.