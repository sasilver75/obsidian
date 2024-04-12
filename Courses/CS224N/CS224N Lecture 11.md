#lecture 
Link: https://www.youtube.com/watch?v=N9L32bFieEY

-----

Today:
1. What is NLG?
2. A review: Neural NLG model and training algorithm
3. Decoding from NLG models
4. Training NLG models
5. Evaluating NLG models
6. Ethical considerations of recent NLG systems (eg [[ChatGPT]])


-------
# What is NLG?
- [[Natural Language Processing]] (NLP) has been split into two broad categories
	- [[Natural Language Understanding]] (NLU)
		- Mostly means that task input is in natural language
		- Semantic parsing, [[Natural Language Inference]], etc.
	- [[Natural Language Generation]] (NLG)
		- The task *output* is in natural language
		- Focuses on systems that produce coherent, fluent, and useful language output for human consumption.
		- [[Machine Translation]] (MT), Summarization, Dialogue Systems, Creative story generation, Data-to-text, Visual description of images


![[Pasted image 20240410232231.png]]


![[Pasted image 20240410232645.png]]
One common way to categorize tasks in NLG is to think about how open-ended they are!
- Things like Machine Translation and Summarization aren't very open-ended -- there's (very nearly) a single correct answer (or something like that)!
- Things in the middle of the spectrum like Task-driven Dialog or ChitChat Dialog have more degrees of freedom ("good, You?" "Thanks for asking -- Barely surviving my homework!" The output space is more diverse!)
- On the other end of the spectrum, there are very open-ended things like Story Generation! ("Tell me a story about three little Pigs!" -- The valid output space is very large!)

Above: ==Open-ended generation vs Non-open-ended generation==
- One way of formalizing a categorization is by the entropy!
- Each of these two classes require different
	- Training strategies
	- Decoding methods


![[Pasted image 20240411124354.png]]
At each timestep, we take in a sequence of tokens as input, and the output is a new token.
- To decide on that output token, we use the model to assign a score to each token in the vocabulary, softmax to get a distribution, and sample a token from this distribution.
- Once we've predicted a token, we then recursively feed that (and the prior sequence) back into the model

![[Pasted image 20240411124530.png]]
Two types of NLG tasks we talked about (Open-ended, Non-open-ended) both tend to use different architectures. For non-open-ended, we often have a [[Encoder-Decoder Architecture]], and for open-ended autoregressive generation models, we often have a [[Decoder-Only Architecture]].
- These aren't *hard constraints*; an autoregressive decoder-only can be used for MT, and an encoder-decoder can be used for story generation. But it's an allocation of resources problem, in terms of compute.

![[Pasted image 20240411124855.png]]
How do we train such a language model?
- We know that they're trained by maximum likelihood; we want to maximize the probability of the next token given proceeding words.
- At each timestep, it's a classification task; we try to distinguish the actual word from all of the other words in the vocabulary.
- This called [[Teacher Forcing]], *because we reset at each timestep to the ground truth*, and try to predict at every timestep the next token. We do teacher forcing when we train the model.
	- At generation time, you don't have any access to the correct next token, so you have to use the model's own prediction, and feed it back into the model to continue generation. This is called [[Student Forcing]]. We don't have access to the gold reference data, but we still want to generate a sequence of data -- so we have to feed our own predictions back into the next generation round.


![[Pasted image 20240411124956.png]]
At inference time, our decoding algorithm defines some function to select a token from this distribution over tokens in our vocabulary.
- Recall, we use the langauge model to compute the next token distribution; we then have to select which token we actually use for $\hat{y}_t$  

Two interesting levers:
1. Improving training 
2. Change how we do decoding

---
# Decoding from NLG Models

![[Pasted image 20240411125148.png]]
What is decoding all about?
- Our model predicts a vector of scores (logits) for each token in our vocabulary
- We then compute a probability distribution P over these scores with a softmax function
- Our decoding algorithm then selects a token from that distribution

We've mentioned [[Greedy Decoding]], where we just select the highest-probability token from the distribution.
We've mentioned [[Beam Search]], where we explore a wider range of candidates by keeping $k$ candidates in the beam that we search though.

==These are good for non-open-ended generation==, but for open-ended generation, the *most likely string* that the above techniques help us generate is actually... a very boring string! Often time sit also begins repeating itself!

![[Pasted image 20240411125648.png]]

Is finding the most likely string reasonable for open-ended generation?
- Probably not -- this doesn't match how humans generate text!

![[Pasted image 20240411125720.png]]
When a human talks, there's a lot of uncertainty, which you can see in the fluctuations in probability; whereas for beam search, it's always generate quite confident generations.


Option: Random Sampling!
![[Pasted image 20240411125952.png]]


![[Pasted image 20240411130104.png]]
Option: [[Top-K Sampling]]
- Vanilla sampling makes every token in the vocabulary an option -- we might end up choosing a very bad word, given the context!
- Solution: Just cut off the (heavy) tail of the distribution (Zipfian distribution?), and only sample from the top tokens in the distribution!
==k is a hyperpararameter==:
1. Increasing k yields more diverse outputs, but riskier change of bad tokens
2. Decreasing k yields more safe outputs, but chance of the generation being boring

![[Pasted image 20240411130458.png]]
Is Top-K good enough? Not really!
- Top-k sampling can cut off *too quickly* because they aren't in the top-k, even though they're good
- Top-k sampling can cut off *too slowly*, and include bad tokens in the top-k!

==The problem above is that the probability distributions we sample from are dynamic!==
- If the probability distribution is flat, having a limited k removes many viable options!
- If the probability distribution is peakier, a high-k allows for too many options to have a chance of being selected!

![[Pasted image 20240411131239.png]]
Introducing: [[Top-P Sampling]] (Nucleus Sampling)
- Sample from all tokens in the top-p cumulative probability mass (i.e. where mass is concentrated)
	- ((EG sample from the top 60% of tokens, however that's distributed!))
	- ==This is equivalent to having an adaptive k for each different distribution!==

This whole idea is not intended to save compute;
- To do top-p or top-k, we still need to compute the softmax over the entire vocabulary set.
So it's not saving compute, it's performing improvement!

Other options:

![[Pasted image 20240411141019.png]]
[[Typical Sampling]] 
- Reweight the score based on the entropy of the distribution
	- (("The entropy... suppose we have a discrete distribution, the negative log probability of x?")) A flat distribution is high-entropy, a peaked is not.
[[Epsilon Sampling]]
- Set a threshold for lower-bounding valid probabilities
- If you have a word whose probability is less than (eg) .03, that word will never be part of your output.

![[Pasted image 20240411141510.png]]
Something else we can tune to effect decoding is the [[Temperature]] parameter!
- When we use Softmax to renormalize our logits into a probabilty... we can insert a Temperature parameter!
- This temperature doesn't affect the monotonicity of the distribution; if word A has higher probability than word B previously, then after the adjustment A will still have a higher than B, but ==their relative difference will change!==
	- Temperature $\tau$ > 1  --> Our distribution becomes more uniform, producing more diverse output!
	- Temperature $\tau$ < 1 --> Our distribution becomes more spiky, producing more "boring" output, since probability is concentrated on top words. In the extreme case, it turns into effectively greedy decoding, producing basically one-hot vectors!

Temperature is a hyperparameter for decoding that can be tuned for either Beam Search or other Sampling methods -- so it's kind of orthogonal to the other sampling examples.

































