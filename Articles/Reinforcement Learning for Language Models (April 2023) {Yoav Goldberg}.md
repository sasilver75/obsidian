#article #premium 
Link: https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81

----

## Why RL?
- With all the talk about [[Reinforcement Learning from Human Feedback|RLHF]], why is RL better than learning from demonstrations (aka supervised learning) for training language models? Shouldn't learning from demonstrations (eg "supervised fine-tuning") be sufficient?
- The author acme to realize that ==there's an argument which not only *supports the case of RL training, but also requires in, in particular with models like ChatGPT!*==
	- This is spelled out in the first half of a [talk](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81) by [[John Schulman]] from OpenAI

This post pretty much repeats his arguments in more words, and adds some things that John didn't explicitly say (but which the author is sure he thought about).


# Background: Supervised learning vs RL
- Let's explain the two learning scenarios so that we're on the same page:

### Pre-training
- In both settings, we assume an LM is first pre-trained over a large body of text, with the objective of predicting the next token. We have a model that, for every sequence of words, can assign a probability over the options of the potential next word. By doing so, it also acquires some sort of internal representation of the language. After doing this process, the model is very capable of generating natural-seeming continuations of text prefixes, but it's not yet good at "communicating" or "solving problems" for us.
- We can craft inputs in a way that their natural continuation will solve our problem (==prompt engineering==), or guide it towards our desired behavior via ==fine-tuning==

### Supervised Training
- In the supervised learning case (==instruction tuning==), we collect a bunch of human-authored texts that have the form of a question/instruction, followed by the desired output. These are our "golden examples" that we'd like the model to emulate.
- These texts might be a question followed by its answer (for example, these texts can be a question followed the answer, or something like `summarize the following text {text}`), followed by a human-authored summary.
- ==The hope is that it would generalize this behavior to other queries, not seen in training.==

### Reinforcement Learning
- Much harder than supervised learning for several reasons:
	1. The [[Credit Assignment Problem]] -- the language model generates a sequence of tokens, and only gets a score at the end of the sequence.
		- This signal is weak; which parts of the answer (i.e. ==which generated tokens) are good, and which are bad?== Which "choices" helped us "get to our goal/reward?"
		- A lot of technical work in RL attempts to solve this problem; it's an active research area with a few reasonable solutions.
	2. We ==need a scoring mechanism to score the answers== (either assigning some scalar score, or comparing two answers) ==to stand in as our reward==.
		- In the context of open-ended language-based tasks, it's ==*hard* to come up with an automatic scorer==, so ==we're left (to start with) with needing human feedback for each generation. This is rather expensive and slow, and is a rather sparse signal (see: credit assignment problem)==

So given these difficulties, why should we choose to use RL, and not just supervised learning?

# The diversity argument
- Perhaps the most intuitive argument against supervised learning/instruction tuning is that we're teaching the learner to replicate the exact answer given by the demonstrator, while ==the reality of human language is that there might be many ways of conveying an equally-good message==, and they might all be valid answers.
- ==In the supervised learning regime, we "punish" the model for even slight deviations from our prescribed text, even if that deviation has the same semantic meaning== -- we might even insist of phrasing which is hard for the model to learn, even if the model already knows how to produce a reasonable, valid alternative answer.
- ==We thus would like the diversity afforded by RL training!==

This seems like an intuitive argument, but not a very convincing one, given that supervised learning *does* seem to work well in practice...and training RL models has many challenges.


# The theoretical argument
- The first "convincing" justification for RL vs supervised learning that the author came up with is that ==supervised learning allows for only positive feedback, while RL also allows for negative feedback.==
	- ==Positive feedback==: We show the model a series of questions and their correct answers
	- ==Negative feedback==: The model is allowed to generate an answer and get feedback saying "this is *not* correct".
- ==From a formal learning perspective, there's a big difference between the two - negative feedback is much more powerful!== 
	- The theoretical argument for this is that roughly, when learning only from demonstrations, an adversarial (or negligent) demonstrator can *mislead* the learner into learning the wrong hypothesis by withholding some important examples. The demonstrator controls the learning process entirely.
	- ==However,== IF, as a learner, you're allowed to form your own hypotheses and ask the teacher if they're correct (as in the RL setting), even an adversarial teacher can no longer trick you into a latching on to a wrong hypothesis! It *must* disclose that it's wrong, if you ask about it -- the learner is now much more powerful!

The author does believe that this is part of the reason that RL is needed, but there's an *additional* argument for why RL might be needed, that's even more important in the context of training LLMs.


# The core argument
- The previous two arguments have relied on hypotheses like "it might be harder for the model to learn," or "a negligent demonstrator that can only give positive feedback/examples might confuse the model" -- both of these *might* hold in practice; in contrast, the following argument provably holds!

- There are (at least) three modes of interaction with a language mode:
	1. ==Text-Grounded==: We provide the model with a *text* and an *instruction*, and expect the model's response to be fully-grounded in the provided text. ("summarize this text," "what are the chemical names mentioned in this text," "translate this text to Spanish")
	2. ==Knowledge-Seeking==: We provide the model with a question or instruction, and expect a (truthful) answer based on the model's internal knowledge ("What are common causes of the flu")
	3. ==Creative==: We provide the model with a question or instruction, and expect some creative output ("Write a story about...")

The argument for RL is based on the interaction type (b), knowledge-seeking queries, in which ==we expect a truthful (or confident) answer, *and* the ability for the model to say "I don't know," or even refuse to answer in situations in which it is uncertain==.

For this type of interaction, we *must* use RL training, because supervised training teaches the model to lie.
- The core issue: ==We want to encourage the model to answer based on its internal knowledge, but we don't *know* exactly what this internal knowledge contains!==

In ==supervised training==, we present the model with a question and its correct answer, and train the model to replicate the provided answer.
In this scenario, there are two cases:
1. The model "knows" the answer
	- In this case, the supervised training correctly pushes it to associate the answer with the question, hopefully pushing it to perform similar steps to answer similar questions in the future.
2. The model "does not know" the answer
	-  In this case,In this case, the supervised training pushes the model to associate the answer with the question anyhow.
	- It may push the model to memorize this particular question-answer pair (this isn't harmful, but it's also not very effective; we want our models to generalize and learn to answer every question, not just the ones in the training data).
	- ==If we succeed in training the model to generalize in these cases, then we essentially teach the model to make stuff up!== This is bad.

((I think it's not true to say that the model can't acquire knowledge during the fine-tuning process... so are we teaching the model new things, or are we teaching it to lie? I'm not sure that it's clear to me... There's "teaching the answer to a question", and there's "teaching generalizable underlying information" -- surely the latter takes more training examples, and diverse ones. But this still feels like intuition.))

==Because we don't know whether the model knows or not, we cannot avoid case 2 above, which is a serious issue for supervised training!== We can't use pure supervised learning to push the model to produce truthful answers, so we must use RL.
- In contrast to the supervised setting, ==the RL setting doesn't encourage the model to lie==: Even if the model does initially guess some answers correctly and learns a "making stuff up" behavior by mistake, in the long run it will get bad scores for made-up answers (which are likely to be incorrect), and learn to adopt a policy that relies on its internal knowledge, or abstain.


## Smaller Remark: Teaching to Abstain
- If a model doesn't know the answer, it should respond with an answer like "I don't know"; this is non-trivial to do.
	- It's hard to do in a ==supervised setting==, because we don't know what the model knows or not.
		- We could probably train it to do things like "never answer questions about people."
	- It's challenging also for the ==RL setting==: The model might not produce "I don't know" answers to *begin with,* so we would have no way of encouraging it to generate such answers.
		- A way around this is to *start* with some supervised training learning to produce "I don't know answers" in some cases, and then continue the process with RL.

# Implications on Model Stealing/Distillation
- OpenAI has invested a lot of effort in RL-type tuning of its LMs.
- There's a recent trend of taking other, publicly-available base language models na training them on GPT examples of GPT outputs, in order to replicate the GPT model's impressive behaviors. (([[Alpaca]], etc.))
	- This is akin to supervised instruction-tuning: Models are trained to produce GPT answers exactly. 
	- This works, well, except it ==doesn't work well for case (b) above, *teaching the model to answer knowledge-seeking queries*==. 
		- The publicly-available base model likely knows a different set of facts from the OpenAI model, and training to replicate GPT's answers will suffer from the same issue supervised training suffers from: ==the model will be encouraged to make up facts to these types of queries, and additionally may learn to abstain in cases where it does know the answer but the GPT model did not.==

# Towards RL without human feedback
- For a long time, training generate language tasks with RL has been impractical for most players -- it required human feedback for every training sample.
	- This is both expensive and extremely slow, especially for models that need to see tens of thousands of examples.
- But RL training now becoming practical, because LMs themselves are paving the way towards removing humans from the RL loop, themselves able to provide effective automatic scoring metrics that can be used in an RL setting: 
	- Train on the human-provided instruction-response pairs, but rather than trying to replicate the human responses directly, let the model generate its own response, and compare the model-generated response to the human provided one using a dedicated text comparison model trained in a supervised fashion ((a reward model)).



