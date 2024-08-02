https://machinelearning.apple.com/research/introducing-apple-foundation-models

See also Note [[Apple Intelligence Foundation Language Models]] (the ~40-page paper released ~1.5 months later on July 29, 2024)

---

Apple Intelligence is comprised of multiple models specialized for users' everyday tasks, that can *adapt on the fly* for the current activity!
- They have been fine-tuned for *user experiences* such as:
	- Writing and refining text
	- Prioritizing and summarizing notifications
	- Creating playful images for conversations with family and friends
	- Taking in-app actions to simplify interactions across apps

We detail how two of these models (3B ==on-device== LM, and a 'larger' ==server-based== LM available with ==Private Cloud Computer== running on Apple silicon servers) have been built and adapted. These two models are part of a larger family of generative models that Apple with share more information on soon.


Apple ==Responsible AI Principles==
1. Empower user with intelligent tools (address specific user needs with responsible tools)
2. Represent our users (global, diversity)
3. Design with Care (misuse, harm)
4. Protect Privacy (on-device and Private Cloud Compute)

![[Pasted image 20240801224547.png|500]]


## Pre-training
- Apple uses their own AXLearn framework building on top of JAX and XLA, and trains models on TPUs and GPUs, with a combination of [[Data Parallelism]], [[Tensor Parallelism]], [[Sequence Parallelism]], and [[Fully Sharded Data Parallelism|FSDP]] to scale training along multiple dimensions.
- Apple trains on licensed data as well as publicly-available data collected by their ==AppleBot== crawler. They use profanity/PII filtering, as well as their own data extraction, deduplication, and ==model-based classification== for document quality.

## Post-Training
- We find that data quality is essential to model success, so we utilize a hybrid data strategy in our pretraining pipeline, ==incorporating both human-annotation and synthetic data==, and conduct thorough data curation and filtering procedures.
- ==We develop two novel algorithms in post-training:==
	1. A [[Rejection Sampling]] fine-tuning algorithm with a teacher committee
	2. A [[Reinforcement Learning from Human Feedback|RLHF]] algorithm with *mirror descent policy optimization* and a leave-one-out advantage estimator.

## Optimization
- We use a range of innovative techniques to optimize them on-device and on our private cloud for speed and efficiency.
- Both on-device and server models use [[Grouped Query Attention]] and [[Weight Tying]] for the input/output embedding layers.
	- on-device model: 49k vocab size
	- server model: 100k vocab size
- On-device model uses *==low-bit palletization==*, a critical optimization technique.
- To maintain model quality, they use a new framework using LoRA adapters that incorporates a mixed 2-bit and 4-bit configuration strategy, averaging ==3.7 bits-per-weight== to achieve the same accuracy as the uncompressed models. The model can even be compressed to 3.5 bits-per-weight without significant quality loss.

==On a iPhone 15 Pro, they get a time-to-first-token latency of about 0.6 milliseconds per prompt token, and a generation rate of 30 tokens per second.==
- This is attained *BEFORE* using any token speculation techniques ([[Speculative Decoding]]), from which we see *further* gains!


## Model Adaptation
- Our foundation models are fine-tuned to users' everyday activities, and can *dynamically specialize themselves on the fly, for the task at hand!* 
	- This is done using ==[[Adapter]]s==, small NN modules that can be plugged into ***various layers*** of the pretrained model.
	- Authors apply adapters to:
		- Attention matrices
		- Attention projection matrix
		- Fully-connected layers
	- ==By finetuning only adapter layers, the original parameters of the base pretrained model remain unchanged, preserving the general knowledge of the model while tailoring adapter layers to support specific tasks.==
- The values of adapter parameters are represented using 16 bits, and for the on-device model, the parameters for an adapter require only *==tens of megabytes==*; ==they can be dynamically loaded, temporarily cached in memory, and swapped== -- ==giving our foundation model the ability to specialize itself on the fly== for the task at hand while efficiently managing memory and guaranteeing operating systems' responsiveness.
	- ((In a sense, this is kind of like a mixture of experts model, except not all experts need to be loaded into memory; only the ones needed are loaded. This isn't an especially accurate comparison, though.))
- To facilitate the training of the adapters, ==we created an efficient infrastructure that allows us to rapidly retrain, test, and deploy adapters== when either the base model or the training data gets updated. The adapter parameters are initialized using the accuracy-recovery adapter introduced in the Optimization section.

![[Pasted image 20240801230443.png|400]]


## Performance and Evaluation
- When benchmarking models, we focus on human evaluation as we find that these results are highly-correlated to user experience in our products.
- Training data (for summarization adapters?) is based on synthetic summaries generated from bigger server models, ==filtered by a [[Rejection Sampling]] strategy that keeps only the high-quality summaries.==
	- ((I wonder what it means here for them to *filter* via rejection sampling? Just that they use a reward model (eg trained on human data?) to filter out bad synthetic summaries?))
- To evaluate product-specific summarization (==since each of {emails, messages, and notifications} summarization requirements differ in different ways==), we use a set of 750 responses carefully sampled/constructed for each use case.


![[Pasted image 20240801233538.png|400]]
Cool that their on-device model (3B) so handily beats Gemma-7B. It's interesting by how much the models' performances vary depending on the use-case (email, message, notification).

In addition to evaluating feature-specific performance, they also evaluate the models' *general capabilities,* using a comprehensive evaluation set of real-world prompts diverse across difficulty levels and categories like ==brainstorming, classification, closed question answering, coding extraction, math reasoning, open question answering, rewriting, safety, summarization, and writing.==
![[Pasted image 20240801233754.png|400]]
==It's again interesting that the on-device model is about as good as [[LLaMA 3]] 8B on "general" evaluation, it seems!== (Note that this figure is before the LLaMA 3.1 models were released)


Apple also uses a diverse set of *adversarial prompts* to test performance on harmful content, sensitive topics, and factuality.
![[Pasted image 20240801233937.png|500]]
It's impressive how harmless these models are. I suppose it remains to be seen how often they're doing false-refusals for borderline topics?
A similar evaluation on safety prompts:
![[Pasted image 20240801234112.png|400]]
Human Preference Evaluation on safety prompts is sort of a weird thing. Because if they both refuse, the human preference is more about... who apologizes better? Which seems like a sycophancy thing, unless you have really disciplined annotators.


![[Pasted image 20240801235525.png]]
See that the Apple 3B on-device model seems to follow instructions better than comparable 8B models, and the Server model better than comparable ~70B+ models... Which *probably* gives us some idea as to the size of the server model (Probably around LLaMA 3 70B size)
- As swyx noted, it's interesting that the only external benchmark that they quote in this blog post is [[IFEval]], an instruction-following eval.

![[Pasted image 20240801235750.png|500]]

## Conclusion
- The Apple foundation models and adapters introduced at WWDC24 underlie Apple Intelligence!



