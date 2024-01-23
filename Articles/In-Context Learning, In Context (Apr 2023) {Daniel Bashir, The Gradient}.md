---
tags:
  - article
---

Link: https://thegradient.pub/in-context-learning-in-context/

-------
Much recent work on [[Large Language Model|LLM]]s has explored the phenomenon of [[In-Context Learning]]. In this paradigm, an LLM learns to solve a new task *at inference time*, without any change to its weights, by being fed a prompt with examples of that task.

Examples
- Translations
- Word corrections
- Arithmetic

The GPT-4 technical report includes examples of questions, answers, and answer explanations when asking the model to answer new questions.

![[Pasted image 20240122185908.png]]
Source: GPT-4 technical report

The brunt of why [[In-Context Learning|ICL]] has been interesting is that, without any further fine-tuning, a pre-trained model is able to learn to do something *entirely new* by merely being shown a few input-output examples.

ICL was defined in the GPT-3 paper, [[Language Models are Few-Shot Learners (May 2020) {OpenAI, GPT-3 Paper}|Language Models are Few-Shot Learners]].

> During unsupervised pre-training, a language model develops a broad set of skills and pattern recognition abilities. _It then uses these abilities at inference time to rapidly adapt to or recognize the desired task_. We use the term “in-context learning” to describe the inner loop of this process, which occurs within the forward-pass upon each sequence.

However the *mechanisms underlying ICL* remain the subject of contending explanations! These attempts at explanation are the subject of this section.


### Early Analyses
- The GPT-3 paper hypothesized that larger models would bring additional gains for ICL abilities. 
- It's worth noting that GPT-3 showed no clear ability to do ICL on the [[Winograd]] dataset, and only mixed results on commonsense reasoning tasks.

![[Pasted image 20240122191943.png]]
Above: ==Larger models are better at ICL==.

Another paper ("An explanation of in-context learning as implicit bayesian inference") attempted to develop a framework for how ICL emerges during training. The authors understand ICL as "locating" latent concepts that the LM has acquired from pre-training data... (skipping some here)

A recent paper named "Rethinking the role of demonstrations: What makes In=Context learning work?" analyzed which aspects of a prompt affected downstream task performance on in-context learners.
- ==The authors ATTEMPTED to identify four aspects of the in-prompt demonstrations that *could* provide learning signal==:
	- The input-label mapping
	- The distribution of the input text
	- The label space
	- The format

In particular, the authors found that ground truth demonstrations are *not required* to achieve improved performance on a range of classification and multi-choice tasks!
- ==In fact, demonstrations with random labels achieved similar improvement to demonstrations with gold labels! And *both* outperformed a zero-shot baseline!==

==The *format* turns out to be very important to improvements from demonstrations==: removing the format, or providing demonstrations with labels only, performs close to or worse than no demonstrations.

==In summary, the *three things that ACTUALLY matter for in-context demonstrations are*==
- The *input distribution* (the underlying distribution inputs in ICL examples come from)
- The *output space* (the set of outputs -- classes or answer choices in the task)
- The *format *of the demonstration

Since the LMs do not rely on the input-output correspondence in a prompt, they might have already been exposed to notions of this correspondence during pretraining that ICL then leverages.


### Induction Heads and Gradient Descent

While studying transformers under the lens of mechanistic interpretability, Anthropic researchers studied a circuit they dubbed an [[Induction Head]]. Studying simplified, attention-only transformers, the researchers found that two-layer models use attention-head composition to create these induction heads, which perform in-context learning...
- In simple terms, Induction Heads search over their context for previous examples of the token that a model is currently processing. If they find it, they look at the *following* token and copy it -- this allow them to repeat previous sequences of tokens to form new completions. Essentially, the idea is similar to copy-psting from previous outputs of the model.


==The upshot is that induction heads appear in [[Transformer]] models *just* about the time that transformers improve in their ability to do ICL!==

Papers make the claim that transformer-based ICLs indeed implement standard learning algorithms *implicitly*; the authors hypothesize that a transformer forward pass implements ICL by "gradient based optimization of an implicit auto-regressive inner loss constructed from its in-context data."

### Learnability and ICL in Larger Language Models
- ...
- ==LLMs demonstrate an *emergent ability* to override their semantic priors and learn from input-label mappings, if they don't totally match.==
	- In other words, a sufficiently large model asked to in-context learn on examples with flipped labels will then show *==degraded performance==* on an evaluation set whose examples have correct labels.


### What's next for ICL?
- Quote from Sewon Min
> I believe that whatever is happening at test time is a consequence of "learning" that happened at pre-training time. This is related to our claim that ==in-context learning is mainly for better LOCATION (activation) of the intrinsic abilities LMs have learned during training==, which has been claimed by other papers as well.

The idea that ICL merely *locates* some latent ability or piece of knowledge that an LM has already imbibed from its training data feels intuitive and makes the phenomenon a bit more plausible.




















