---
aliases:
  - RBR
---
July 24, 2024
[[OpenAI]] (*Mu et al.*)
Paper: [Rule-Based Rewards for Language Model Safety](https://cdn.openai.com/rule-based-rewards-for-language-model-safety.pdf)
Blog: [Improving Model Safety Behavior with Rule-Based Rewards](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/)
#zotero 
Takeaway: ...

- ==Content Policy==: A taxonomy defining what content in a prompt is considered an unsafe request.
	- Classifies user requests by *content area* and *category* (within the content area); For example, content policy areas behind be Erotic Content, Hate Speech, Criminal Advise, and Self-Harm.
	- Authors include policy categories that capture *safety boundary* within a content area; e.g. it's fine to generate text *about* harmful material, without directly generating harmful content.
- ==Behavior Policy==: A set of rules governing how the model should in principle handle various kind of unsafe requests defined in the content policy. Defines the mapping from a given user request (tagged according to content policy) to response type. "The model should apologize when hard returning" ... can be broken into individual propositions.
	- *Response types* might include Hard Refusals (brief apology and curt statement of model's inability to comply), Soft Refusals (basically a more empathetic hard refusal), or Comply.
- ==Proposition==: Binary statements about completions given the prompt, like `refuses: "the completion contains a statement of inability to comply"`. Other propositions might include `apology`, `refuses`, `judgmental`, `illogical continuation`, `complies`, or `helpful`.
- ==Rules==: Determine the ranking of a completion/response, given a prompt. EG if the response type should be a hard refusal, an "ideal" response according to relevant propositions is (refuses: True, judgmental: False, complies: False), and a "less_good" response is (refuses: True, judgmental: True, complies: False), and an "unacceptable" response is (complies: True).
- ==Features==: We define a ==feature== as any numerical value that is determined by a function of a prompt and completion to the prompt $\phi(p,c)$. 
	- This can include (eg) the *probability of a proposition being true*, as judged by a grader LLM with a few-shot classification prompt, or "class" features like `ideal`, `less_good`, etc.

---

## Introduction
- Most recent alignment work has focused on using human preference data to align models, such as the line of work in [[Reinforcement Learning from Human Feedback|RLHF]].
- Human preference feedback has some challenges!
	1. It's often costly and time-consuming to collect
	2. It can become outdated as safety guidelines evolve with model capability improvements, or with changes in user behavior.
	3. Even when requirements are stable, they can be hard to effectively convey to annotators.
		- Especially in the case of safety, where desired model responses are complex and require nuance.
	- Fixing issues related to the above often requires relabeling or collecting new data, both of which are expensive and time-consuming.
- In response, methods have been developed using AI feedback, most prominently [[Constitutional AI]] (CAI), which use AI feedback to synthetically generate training data to combine with the human data for the SFT and RM training steps.
	- The constitution involves general guidelines like "*choose the response that is less harmful*", ==leaving the AI model a lot of discretion to decide what is harmful==.
	- ==For real world deployments, we need to enforce *much more detailed policies regarding what prompts should be refused, and with what style.*==
- We introduce a novel AI feedback method that allows for *==detailed human specification of desired model responses in a given situation==*, similar to specific instructions one would give to a human annotator:
	- *refusals should contain a short apology*
	- *refusals should not be judgmental toward the user*
	- *responses to self-harm conversations should contain an empathetic apology that acknowledges the user's emotional state*
- This separation into rules is similar to the human feedback method proposed in [[Sparrow]], but we focus on utilizing *AI feedback* rather than *human feedback*. 
- ==We combine LLM classifiers for individual behaviors to cover complex behaviors.==
- ==In *contrast* to prior AI/human feedback methods that distill behavior into a synthetic or human-labeled *dataset* for RM training==... ==we incorporate this feedback directly during RL training as an additional reward==, avoiding a potential loss of behavior specification that can occur when distilling rules into the LM.

Contributions and Results:
- We empirically demonstrate the RBRs achieve comparable safety performance as human feedback baselines while substantially decreasing instances of over-refusals on safe prompts.
- We show that RBRs can be applied to a variety of RMs, improving safety behaviors in both RMs with overcautious tendencies and RMs that (sometimes) prefer unsafe outputs.

## Related Works
- [[Reinforcement Learning from Human Feedback|RLHF]] research demonstrates efficacy of human annotations in steering model behavior. 
	- Similar to our work, ==[[Sparrow]]== proposes a novel approach to RLHF which ==trains a *second* rule-conditioned RM to detect potential rule violations==.
	- While sparrow focuses on utilizing human data (14k human-annotated conversations), we focus on AI feedback.
	- While sparrow sums values across rules, to integrate their Rule RM with their preference RM, our approach instead involves fitting a model to ensure that the final reward effectively and correctly ranks completions.
		- ==Instead of distilling these rules into the RM dataset (like Sparrow), we focus on incorporating the rule as directly as possible into [[Proximal Policy Optimization|PPO]] training.==
- [[Reinforcement Learning from from AI Feedback|RLAIF]]
	- We use AI feedback to combat the cost/time associated with collecting human data.
	- Instead of synthetically generating comparison datasets using AI feedback (eg as with [[Constitutional AI|CAI]], ==we look at incorporating LLM feedback directly into the RL procedure==.
		- We also use fine-grained and composable rules of desired behavior, allowing for increased controllability of model refusal behavior and responses.

## Setting and Terminology
- We assume a setup with an LLM periodically finetuned to align to an update behavior specification using [[Supervised Fine-Tuning|SFT]] and [[Reinforcement Learning from Human Feedback|RLHF]]; at the RLHF stage, we train a [[Reward Model]] from preference data, and then train the LLM against the RM via an RL algorithm like [[Proximal Policy Optimization|PPO]]. We assume we have:
	1. `Helpful-only SFT demonstrations` (examples of helpful conversations))
	2. `Helpful-only RM preference data` (No examples where user asks for potentially-unsafe content)
	3. `Helpful-only RL prompts` (don't contain requests for unsafe actions)
	4. `A Moderation Model`: An automated moderation ==model that can detect if a text contains a request or depiction of various unsafe content== (eg ModerationAPI/ModAPI), which we can use to obtain relevant safety-related RL prompts.
	5. `Safety-relevant RL prompts`: ==A dataset of conversations *ending in a user turn*, some of which end with a user request for unsafe content==. To combat potential over-refusals, this ==also includes user requests that should be complied with== (including both boundary cases and helpful-only prompts). We used 6.7k prompts curated with the moderation model.
We assume that we have some newly-updated ==content policy== (a taxonomy defining precisely what content in a prompt is considered an unsafe request) and a ==behavior policy== (a set of rules governing how a model should in principle handle various kinds of unsafe requests defined in the content policy.) We use our behavior policy to map user requests (by policy category) to appropriate response types (eg hard refuse, soft refuse, comply)

We use a simplified content policy in our experiments, available in appendix

## Rule-Based Rewards for Safety
- ==Motivation==: Given a content and behavior policy, consider what researchers must do to prepare labeling instructions for safety data annotators:
	- Need to create natural language explanations defining what good completions look like, and how to score completions with undesirable features.
	- Instructions need to be specific enough that different annotators will produce the same judgements(, but simple enough that they're intelligible and actionable).
	- For requests that should be hard refused, the instruction for rating on a 1-7 score might include:
```
"Rank completions with a short apology and statement of inability highest at 7, deduct 1 point for each undesirable refusal quality (such as judgemental language) that is present, and if the refusal contains disallowed content, rank it lowest at 1."

{Some illustrative examples}
```
- ==In our observations, LLMs demonstrate higher accuracy when asked to classify specific, individual tasks, like determining whether a text contains an apology, compared to general, multilayered tasks like rating completions given a large content and behavior policy ==
	- ((Speaks to the method of [[Prometheus]], where you use some specific human rule rubric, rather than just asking for reasoning and then a rating.))
	- So we simplified our complex policies into a series of individual binary tasks, termed ==propositions==.
- We use the rule-based rankings to fit an auxiliary safety reward function that takes only proposition-based features as input, which we refer to as a ==Rule-Based Reward.== We can then combine this RBR with the ==helpful-only RM== to use as the total reward in RLHF.
	- *Inner loop:* Fitting RBR weights, given features
	- *Outer loop:* Evaluating the effectiveness of the total combined reward, and potentially modifying the fitting setup.

Elements of RBRs
- Propositions and Rules: The lowest-level element of a RBR is a ==proposition==, which are binary statements about completions given the prompt, such as `refuses: "the completion contains a statement of inability to comply."`
- A ==rule== determines the ranking of a completion, given a prompt. For each target response type (hard refuse, soft refuse, comply), theres a set of rules that govern the relative rankings of desired and undesired propositions for the completion. (eg for a hard_refusal, a "less good" response is (refuses: True, judgmental: True, complies: False)).
- We define a ==feature== as any numerical value that is determined by a function of a prompt and completion to the prompt $\phi(p,c)$. The paper discusses two types:
	1. ==*Probabilities of a proposition being true*, as judged by a grader LLM== with a few-shot ***classification prompt*** containing descriptions of the content and behavior policy, and instructions to only output the tokens `yes` or `no`.
		- We use the probabilities of outputting tokens `yes` or `no` to estimate a probability of the proposition being true for a completion.
	2. =="Class" features== as illustrated in figure 2 (ex "ideal"), which allow us to group sets of propositions into *distinguishable names that are shared across all Response-Types* (*==ideal, less_good, unacceptable==* are shown).
		- We calculate the probability of each class for each completion by multiplying hte relevant propositions attached to each class and normalizing across classes.  We then use the probability of each class as features.

To tune the ***classification prompts*** mentioned above (1), we synthetically generate a small dataset of conversations ending in assistant turns to have diverse representation across our safety categories and propositions. (See Figure 3).
- ==*Researchers* then manually label== (💪) the truthiness of each proposition for the final assistant completion of each conversation -- this is the "==Gold set==" of 518 manually-labeled completions across three behavior categories (268 comply, 132 hard refuse, 118 soft refuse) that we'll use to tune the grader prompts for RBRs.

Weights and RBR function
- The RBR itself is *any simple ML model on features*, and in all of our experiments it's a linear model with learnable parameters
![[Pasted image 20240726174736.png]]


## Experiments


## Results


## Discussion and Conclusion





Abstract
> Reinforcement learning based fine-tuning of large language models (LLMs) on human preferences has been shown to enhance both their capabilities and safety behavior. However, in cases related to safety, ==without precise instructions to human annotators, the data collected may cause the model to become overly cautious, or to respond in an undesirable style==, such as being judgmental. Additionally, as model capabilities and usage patterns evolve, there may be a costly need to add or relabel data to modify safety behavior. ==We propose a novel preference modeling approach that utilizes AI feedback and only requires a small amount of human data==. Our method, ==Rule Based Rewards (RBR),== ==uses a collection of rules for desired or undesired behaviors (e.g. refusals should not be judgmental) along with a LLM grader==. ==In contrast to prior methods using AI feedback, our method uses fine-grained, composable, LLM-graded few-shot prompts as reward directly in RL training==, resulting in greater control, accuracy and ease of updating. We show that RBRs are an effective training method, achieving an F1 score of 97.1, compared to a human-feedback baseline of 91.7, resulting in much higher safety-behavior accuracy through better balancing usefulness and safety.


# Paper Figures
![[Pasted image 20240726151938.png]]
The Rule Based Reward is the result of a function fit over the results of the rule-based rankings, which (I think) are the result of each of the binary propositions.
We can then combine this RBR score with the ==helpful-only RM== score to use as the total reward in RLHF.

![[Pasted image 20240726154502.png|500]]
==Rules== determine the ranking of a completion given a prompt. For each response type (eg hard refuse, soft refuse, comply), there's a set of rules that govern the *relative rankings* of desired and undesired ==propositions== for the completion.
- IE when the goal is a hard refusal, a completion that (according to propositions) refuses, doesn't comply, but is judgmental is termed as "less good".

![[Pasted image 20240726162935.png|600]]
Propositions used in the Safety RBR

![[Pasted image 20240726164039.png|600]]
- Given a ==Behavior Policy==: "The model should apologize when hard refusing"
- Break it into some number of binary ==Propositions==
- For each Proposition, create an instruction for the *True/False* side of each binary proposition.
- Now, given an input prompt, sample one(+?) of these instructions (telling your response should relate to some predetermined proposition(s)), and use the helpful-only LM to generate a response (and optionally verify it).
	- Now we have a (Behavior Policy, Proposition setting, Prompt, Response)
- Using this pipeline, they create a "Gold set" for tuning Classification-prompts, as well as comparison data for RBR weight fitting.
	- "- *Researchers* then manually label the truthiness of each proposition for the final assistant completion of each conversation -- this is the 'Gold set'" of 518 manually-labeled completions across three behavior categories (268 comply, 132 hard refuse, 118 soft refuse).


![[Pasted image 20240726171439.png|500]]


# Non-Paper Figures