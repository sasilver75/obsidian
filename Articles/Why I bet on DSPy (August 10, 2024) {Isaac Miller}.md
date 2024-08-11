https://blog.isaacmiller.dev/posts/dspy
-----

DSPy is an open-source framework helping you compose multiple LLM calls together in a *principled manner* to solve a problem.

LLMs calls chained together without evaluations are unsustainable for trying to solve real-world problems in the long term.
- How do you know if you're editing prompts in the right direction?
- How do you know if you're actually solving the problem?
- What if you data distribution changes?
- What if you want to edit your pipeline slightly?

==DSPy forces you to use verifiable feedback to solve your problem.==
- This could be a comparison to ground-truth (eg MC, Classification, LM Judge)

LMs are not able to do reasoning in a traditional sense, but they're really good at pattern matching, matching a distribution, and being creative.

When people think about ==automatic prompt optimization==, they often think of systems where LLMs critique, improve, argue, and give feedback to improve the prompt.
- ==This is generally the wrong way to use LLMs!==

When DSPy does prompt optimization: 
- It gives an LLM the current prompt and says:
	- "Hey, come up with creative variations of this that might solve the prompt better."
- Then it actually tries those variations on your evaluation set to see if they solve the problem better. 
- If they don't numerically do better on whatever metric you care about, the ideas are thrown away
	- ((This reminds me of the idea that LMs are good generators, but not validators or critiquers))
- ==It almost looks like an evolutionary algorithm where the prompts are evaluated for their fitness by testing them against your evaluation set.==

So there is ==no deductive LLM reasoning involved==, and there is ==no textual gradients== -- we just harness the strengths of LLMs as *creative engines* (and then use an external way of evaluating fitness).

*==Be skeptical when people tell you that LLMs are giving feedback and doing reasoning!==*
- Do not get fooled into thinking that just because a bunch of LLMs throw nonsense at eachother for millions of tokens, anything valuable can be produced *without verifying it in the real world!*

What now?

1. Generally, you should define systems in terms of inputs, outputs, the steps it takes to get there, and how you know you've done a good job?
	- DSPy forces you to do that!
2. ==LLMs are not good at reasoning and should not be used as such==. If you need a reasoning step, try to verify and connect it to the real world as much as possible.
	- ==Use LLMs as *creative engines* -- generate ideas!==
	- ==Then *cull them down using other methods!*==


What's currently wrong with DSPy?
- The framework isn't wholly reliable! 
	- Quirks like newest pipeline operators that don't work with the Assertions features make it hard to trust new features.
	- Older optimizers are consistent, but many of the newer features are less documented or have bugs.
- We are shifting our focus to fixing this experience across the framework and making DSPy consistently reliable.
- Reliability and approachability are the framework's two biggest flaws.




















