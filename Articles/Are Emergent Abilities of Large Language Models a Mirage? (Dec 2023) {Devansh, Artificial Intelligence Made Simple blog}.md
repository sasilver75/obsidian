#article 
Link: https://artificialintelligencemadesimple.substack.com/p/are-emergent-abilities-of-large-language

------


## Executive Summary
- ==Emergence== refers to the phenomenon where LLMs display abilities not present in smaller scale models.
	- Abilities are said to be emergent when they appear as the model scales to larger sizes (params, data, computational power). The abilities can range from improved problem-solving skills to the generation of more complex and nuanced text.
- These abilities seem to appear unpredictably; as models scale, some metrics show smooth, continuous predictable changes in model performance, whereas other metrics show discontinuous, nonlinear jumps in performance.

![[Pasted image 20240422161012.png]]

The Mirage paper (2023 NIPs best paper) highlights the importance of studying our metrics, and taking a lot of claims about LLMs with a giant pinch of salt. 
When it comes to deploying models, fingertip-feel, good design, and judgement mean a lot more than performance on metrics.

![[Pasted image 20240422161304.png]]

# Introduction
- As we scaled up GPT and Gemeni, we saw sudden, unexplainable jumps in performance on a variety of benchmarks. Researchers calls this phenomenon emergence, after the biological phenomenon where most complex forms of life manifest new abilities that aren't present in simpler organisms.
- AI emerge is a huge cause of concern and hype, because it inherently means that we're likely to create and deploy systems that we're unable to fully analyze, control, or predict.

Some researchers at Stanford decided to pour water on the idea of AI emergence with their Mirage paper, which showed that emergence may not be an inherent property of scaling models, but rather a measurement error caused by inferior measurement selection.
Let's go over three sets of experiments to understand implications in more detail!

# Experiment Set 1: Analyzing InstructGPT/GPT-3's Emergent Arithmetic Abilities
- One of the most eye-catching examples of emergence was AI Arithmetic. Language models that can perform computations perfectly are fantastic for handling tabular data, or for parsing through documents for higher-level retrieval.
...
“_This confirms the first prediction and supports our alternative explanation that the source of emergent abilities is the researcher’s choice of metric, not changes in the model family’s outputs. We also observe that under Token Edit Distance, increasing the length of the target string from 1 to 5 predictably decreases the family’s performance in an approximately quasilinear manner, confirming the first half of our third prediction_.”

# Experiment Set 2: Meta-Analysis of Emergence
- Authors look into the published results of other models that claim emergence, attempting to prove:
	- If emergence is real, one should expect task-model family pairs to show emergence for al l reasonable metrics.
	- On individual task-metric-model family triplets that display an emergent ability, changing the metric to a linear/continuous metric should remove the emergent ability.

Changing the metric when evaluating task-model family pairs causes emergent abilities to disappear.

This is a fairly strong argument against emergence. To tie this up, the authors show us that it is possible to do the ==reverse- to show emergence in vision models by simply changing the metrics we use for evaluation==. This was my fav part b/c thought of some influencer taking this part out of context and claiming that vision models now have emergent abilities made me smile.

# Experiment Set 3: Inducing Emergent Abilities in Networks
- So far, no one has claimed AI emergence in Computer Vision -- but if we can show that simply changing metrics used to evaluate well-known vision models on standard tasks, it would be a strong argument for the thesis of the paper: That ==emergence depends on the choice of metrics more than on a magical property in scaling==.
- The authors demonstrate their ability to do this on autoencoders, autoregressive transformers, and CNNs.
