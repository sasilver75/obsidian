https://www.youtube.com/watch?v=QL-FS_Zcmyo&list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&index=6&t=314s

===

![[Pasted image 20241017221242.png]]

Instead of just taking text as inputs and text as outputs, we use a LLM as the key brain for reasoning/planning for agents that let agent observe and take actions in the environments, using external tools and databases/knowledge bases to help the agent perform tasks.

Rich capabilities of LLMs make LLM agents flexible

![[Pasted image 20241017221425.png]]

Through the interaction with complex/diverse environments, they can update their memory... and obtain grounding through this interaction data.

Agents can interact with other agents through multiagent interactions and collaboration...

So why are Agents the next frontier?
- Trial and error process
- Leverage of external tools/revtrieval to extend capabilities
- Dynamic agentic flow... can facilitate solving complex tasks through task decomposition, allocation of subtasks to specialized submodules, multiagent collaboration, etc.

![[Pasted image 20241017221802.png]]
Mentions [[SWE-bench]], [[GAIA]], [[WebArena]] as relevant benchmarks.

To better enable agent deployments there are a number of key challenges to address:
1. Improve reasoning and planning capabilities of agents
2. Embodiment and learning from environment feedback -- how can we recover from mistakes for long-horizon tasks? (Multimodal learning, etc.)
3. Multi-agent learning, theory of mind of other agents?
4. Safety and privacy (LMs are susceptible to adversarial attacks)
5. Human-agent interaction, ethics (how to control agent behavior)

Topics in Course:
1. Model core capabilities (Reasoning, Planning, Multimodal)
2. LLM agent frameworks (Workflow design, tool use, RAG, multi-agent)
3. Applications (software dev, workflow automation, multimodal, enterprise)
4. Safety and ethics 


----

Denny Zhou @ GDM

What do we expect from AI?

What is missing from ML? Reasoning.
- Humans can learn from just a few examples, *because humans can reason.*

Let's start with a toy problem...

Last Letter Concatenation
![[Pasted image 20241017222812.png]]

This is such a simple task! 
A few years ago, you'd try to solve this using an Encoder/Decoder transformer, and find that... you'd need many labeled examples to train the model.
If the method requires a vast amount of labeled data to learn... An intelligent model should be able to learn this task using very few examples.
How can we solve using LLMs?

We can use [[Few-Shot Prompting]]
![[Pasted image 20241017223239.png]]
In this example, the output is incorrect.

![[Pasted image 20241017223408.png]]
Adding in a "reasoning" process seems to give a correct example, even with just one demonstration (not pictured).

![[Pasted image 20241017223521.png]]
this idea of deriving a final answer through some intermediate steps isn't a new one, and predates [[Chain of Thought]]. In the 2017 DeepMind paper, they used natural language rationale to solve math problems.

In the [[GSM8K]] dataset from OpenAI, which include natural language rationales in the dataset. This was used to finetune [[GPT-3]].
![[Pasted image 20241017223748.png]]

In the same year, 2021, researchers at GDM did a Show Your Work paper, where they let language models use a "scratchpad" for intermediate computation with language models.
![[Pasted image 20241017223825.png]]

![[Pasted image 20241017223847.png]]
In the [[Chain of Thought|CoT]] work, they evaluated the improvement in given by asking models to think step by step, etc.

Overall timeline
![[Pasted image 20241017223922.png]]
So which part is more important?
- Actually it doesn't matter if you train/finetune/prompt the model. What really matters is the *intermediate steps -- that's the key.*

In addition to intermediate steps, is it helpful to e xplicitly introduce reasoning ==strategies== in examples of solving problems?
In [[Least-to-Most Prompting]] (ICLR 2023), they enabled eeasy-to-hard generalization by decomposition.
![[Pasted image 20241017224153.png]]
It seems to me that the bold is a breakdown of the problem in a coarse level of granularity, yielding subproblems. and then they solve each subproblem step by step.


Simple idea, but surprisingly powerful! So there's a SCAN task for compositional generalization...
- Given a natural language command, translate it to a sequence of actions htat could be executed by a robot.
![[Pasted image 20241017224335.png|500]]

In another task, CFQ, which is a text-to-code task....
The idea is that the test examples are more difficult than the training examples. For the text-to-code problems, for the test problems, it requires generation of a larger code snippet, for instance.
![[Pasted image 20241017224421.png]]

... internet died, continue