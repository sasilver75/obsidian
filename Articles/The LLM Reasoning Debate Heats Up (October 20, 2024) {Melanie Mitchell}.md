Link: https://aiguide.substack.com/p/the-llm-reasoning-debate-heats-up?r=68gy5&utm_medium=ios&triedRedirect=true

From [[Melanie Mitchell]]

----

A fierce debate: can language models reason?

OpenAI's GPT-4o release: "... can reason across audio, vision, and text in real time."
OpenAI's o1 release: "...perform complex reasoning, which achieves record accuracy on many reasoning-heavy benchmarks."

Other have questioned the extend to which LLMs solve problems by reasoning abstractly, or whether their success is due to matching reasoning patterns memorized from their training data, which limits their ability to solve problems that differ too much from what has been seen in training.

Does it matter whether LMs reason, or if they perform behavior that just *looks* like reasoning?
> *> Why does this matter? If robust general-purpose reasoning abilities have emerged in LLMs, this bolsters the claim that such systems are an important step on the way to trustworthy general intelligence.  On the other hand, ==if LLMs rely primarily on memorization and pattern-matching rather than true reasoning, then they will not be generalizable—we can’t trust them to perform well on ‘out of distribution’ tasks==, those that are not sufficiently similar to tasks they’ve seen in the training data.* - [[Melanie Mitchell]]

But wait, what IS reasoning anyways?
- It's an overloaded term that can carry quite a few different meanings.

> *The word ‘reasoning’ is an umbrella term that includes ==abilities for deduction, induction, abduction, analogy, common sense, and other ‘rational’ or systematic methods for solving problems==. Reasoning is often a process that involves ==composing multiple steps of inference==. Reasoning is ==typically thought to require abstraction==—that is, the capacity to reason is not limited to a particular example, but is more general. If I can reason about addition, I can not only solve 23+37, but any addition problem that comes my way. If I learn to add in base 10 and also learn about other number bases, my reasoning abilities allow me to quickly learn to add in any other base.*

It's true that systems like GPT-4 and GPT-o1 have excelled on benchmarks that are billed as being "reasoning" benchmarks, but does this mean that they're actually doing this kind of abstract reasoning?
- These reasoning tasks are often similar (or even identically) to ones that were in the model's training dataset.

There have been many papers that explore these hypotheses.
- Most of these test the ==robustness== of LLM reasoning capabilities ==creating superficial variations of tasks that LLMs do well on -- variations that don't change the underlying reasoning required==, but that are less likely to have been seen in the training data.

Here are three different recent papers that Melanie found particularly interesting:

## 1) The embers of autoregressive training
- Paper: [Embers of autoregression show how large language models are shaped by the problem they are trained to solve](https://www.pnas.org/doi/10.1073/pnas.2322420121)
- This is one of Melanie's favorite LLMs papers -- it asks if the way LLMs are trained has lingering effects ("embers") on their problem solving abilities.

Consider the task of reversing a sequence of words: There are two sequences:
> time, the of climate political the by influenced was decision This

> letter. sons, may another also be there with Yet

Getting the right answer shouldn't depend on the particular words in teh sequence, but the authors show that for GPT-4 there's a strong dependence!

Note that the first sequence reverses into a coherence sentence: "This decision was influenced by the political climate of the time," and the second sequence does not: "Yet with there be also another may sons, letter."

We would think that this fact wouldn't impact the ability of the language model to reverse the sequence, but the language model is actually much more likely to reverse the first sequence, rather than the second (incoherent) one!
- GPT-4 gets a 97% accuracy when the answer is a high-probability sequence
- GPT-4 gets 53% accuracy for low-probability sequences

Another of the tasks that authors used to study these sensitivities is by "shift ciphers" -- also called "Rot-n" ciphers, where *n* is the number of alphabetic conditions to shift (rotate) by.

They tested GPT-3.5 and GPT-4 on ciphers of different n's:

Prompt example:

> *Rot-13 is a cipher in which each letter is shifted 13 positions forward in the alphabet. For example, here is a message and its corresponding version in rot-13.*
> 
> *Original text: "Stay here!"*
> 
> *Rot-13 text: "Fgnl urer!"*
> 
> *Here is another message. Encode this message in rot-13*: 
> 
> *Original text: "To this day, we continue to follow these principles."*
> 
> *Rot-13 text:* 

Authors found that GPT models have a strong sensitivity to input and output probability.

In short, "Embers of Autoregression" paper is somewhat of an "evolutionary psychology" for LLMs. It shows that the way LLMs are trained leaves strong traces in the biases that the models have in solving problems.

Results:
> First, we have shown that ==LLMs perform worse on rare tasks than on common ones==, so we should be cautious about applying them to tasks that are rare in pretraining data. Second, we have shown that ==LLMs perform worse on examples with low-probability answers than ones with high-probability answers==, so we should be careful about using LLMs in situations where they might need to produce low- probability text. Overcoming these limitations is an important target for future work in AI.


## 2) Factors affecting "chain of thought" prompting
- Paper: [Link](https://arxiv.org/abs/2407.01687)
- This paper shares two authors with the previous paper, and looks in-depth at Chain-of-Thought (CoT) prompting on the shift-cipher task.
- [[Chain of Thought|CoT]] prompting has been *claimed* to enable robust reasoning in LLMs.
- In (few-shot) CoT prompting, the prompt includes an example of a problem, as well as the reasoning steps to solve it, before posing a new problem:

> Rot-13 is a cipher in which each letter is shifted 13 positions forward by the alphabet.
> For example, here is a message written in rot-13 along with the original text that it was created frmo:
> Rot-13 text: "fgnl"
> To decode this message, we shift each letter 13 positions backwards:
> 1. f -> s
> 2. g -> t
> 3. n -> a
> 4. l -> y
> Therefore, the original text is: "stay"
>
> Here is another message in rot-13. Decode this message one letter at a time. On the last line, write the words: "Original text:": followed by the decoded message:
> Rot-13 text: <test_imput>

The authors saw that without CoT, GPT-4. Claude 3.0, LLaMA 3.1 ==got close to zero accuracy for most shift levels (n), but when using prompts with CoT like the one above, they achieve much higher accuracy== (eg 32% for GPT-4) across shift levels.

The authors cite four possible ways LLMs can *appear* to be "reasoning," each of which makes different predictions about its pattern of errors:
1) ==Memorization==: The model is repeating reasoning patterns memorized from training data. e.g. for shift ciphers, Rot-13 is more frequent than other Rot-n values.
2) ==Probabilistic Reasoning==: The model chooses output that is most probable, given the input... influenced by the probability of token sequences during training.
3) ==Symbolic Reasoning==: Model is using deterministic rules that work well for every input.
4) ==Noisy Reasoning==: The model is using an approximation to symbolic reasoning in which there is a chance of making an error at each step of inference. This would predict that problem that require more inference steps should produce worse accuracy.

To cut to the chase, the authors found that LLMs with CoT prompting exhibit a ==MIX of memorization, probabilistic reasoning, and noisy reasoning.==

![[Pasted image 20241113194913.png|400]]


## 3) Testing the robustness of LLMs on variations of simple math word problems
- Paper: [GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models](https://arxiv.org/abs/2410.05229)
- From a research group at [[Apple]], tests the robustness of several LLMs on a reasoning benchmark consisting of grade-school math word problems ([[GSM8K]]).
	- GSM8K performance might indicate memorization, rather than robust reasoning abilities.
	- ==Authors take problems and create many variations on it, by changing names/numbers/other superficial aspects of the problem -- changes that don't affect the general reasoning required.==
	- They test several LLMs on this variation, and find that in all cases, the model's accuracy decreases from that on the original benchmark -- in some cases by a lot, though ==on the best models, like GPT-4o, the decrease is minimal.==
		- Still, even the best models are remarkably susceptible to being fooled by such additions:
![[Pasted image 20241113195616.png|500]]
Above: Showing the accuracy degradation when we perform the GSM swap.

This paper, released just a couple of weeks ago, got quite a lot of buzz in the AI / ML community.  ==People who were already skeptical of claims of LLM reasoning embraced this paper as proof that [“the emperor has no clothes”](https://x.com/c___f___b/status/1845361693271920826),== and called the GSM-NoOP results [“particularly damning”](https://garymarcus.substack.com/p/llms-dont-do-formal-reasoning-and).

People more bullish on LLM reasoning argued that the paper’s conclusion—that current LLMs are not capable of genuine mathematical reasoning—was too strong, and [hypothesized](https://x.com/boazbaraktcs/status/1844763538260209818) that current LLMs might be able to solve all these problems with proper prompt engineering. (However, I should point out, when LLMs _succeeded_ on the original benchmark without any prompt engineering, many people cited that as “proof” of LLMs’ “emergent” reasoning abilities, and they didn’t ask for more tests of robustness.)

==Others questioned whether humans who could solve the original problems would also be tripped up by the kinds of variations tested in this paper==.  Unfortunately, the authors did not test humans on these new problems.  I would guess that many (certainly not all) people would also be affected by such variations, but perhaps unlike LLMs, we humans have the ability to overcome such biases via careful deliberation and metacognition.  But discussion on that is for a future post.

# Conclusion

==In conclusion, there’s no consensus about the conclusion==!  There are a lot of papers out there demonstrating what looks like sophisticated reasoning behavior in LLMs, but there’s also a lot of evidence that these LLMs aren’t reasoning _abstractly_ or _robustly_, and often over-rely on memorized patterns in their training data, leading to errors on “out of distribution” problems.  Whether this is going to doom approaches like OpenAI’s o1, which was directly trained on people’s reasoning traces, remains to be seen.  In the meantime, I think this kind of debate is actually really good for the science of LLMs, since it spotlights the need for careful, controlled experiments to test robustness—experiments that go far beyond just reporting accuracy—and it also deepens the discussion of what _reasoning_ actually consists of, in humans as well as machines.













