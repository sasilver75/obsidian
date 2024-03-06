#article 
Link: https://cameronrwolfe.substack.com/p/tree-of-thoughts-prompting
Read: Mar 5, 2024

Related Post:
[[Chain of Thought Prompting for LLMs (Apr 2023) {Deep Learning Focus Newsletter}]]

--------

![[Pasted image 20240305230526.png]]

As LLMs first started to gain in popularity, they were criticized for their shortcomings in solving complex, reasoning-based problems. Although scaling up these models provided a near-uniform boost in performance, we saw virtually no boost in performance on reasoning-based tasks with modern LLMs.

This changed with the proposal of advanced prompting techniques, such as [[Chain of Thought]] prompting and [[Self-Consistency]] -- these methods showed us that LLMs are capable of "reasoning" and solving complex, multi-step problems!

> *"It's perhaps surprising that underlying all this progress is still the original autoregressive mechanism for generating text, which makes token-level decisions one by one and in a left-to-right fashion."*

In the [[Tree of Thought]] prompting approach, we solve problems by explicitly decomposing them into a series of *thoughts*, or intermediate steps.
- Similar to [[Chain of Thought]] prompting, ==tree of thoughts prompting generates a solution that is simply a sequence of individual thoughts.==
- However, this approach goes further by ==allowing multiple reasoning paths to be considered at once== -- *forming a tree of potential thoughts or reasoning paths* -- and explores this entire solution space via LLM-powered self-evaluation.
- With Tree of Thoughts, the LLM can deliberately plan its solution, test various intermediate reasoning paths, and even perform backtracking -- allowing the model to explore the solution space and eventually generate the correct output.


## Connections to Research in Other Fields and Generations

> *A genuine problem-solving process involves the repeated use of available information to initiate exploration, which discloses, in turn, more information until a way to attain the solution is finally discovered."*

Humans are sometimes said to have two separate modes of making decisions:
1. A fast, automatic, unconscious mode ("System One thinking")
2. A slow, deliberate, conscious mode ("System Two thinking")

Authors argue that techniques like chain-of-thought prompting seem to mimic the first mode outlined above, as the LLM just generates text in a left-to-right manner without deliberate planning or deconstruction of the problem.

The goal of Tree of Thought prompting is to inject deliberate *planning and exploration* into the problem space, to mimic the second type of thinking.


# The Basics of Prompting



































