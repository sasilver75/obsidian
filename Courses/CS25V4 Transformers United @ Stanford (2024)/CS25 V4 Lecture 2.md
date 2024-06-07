Date: April 11, 2024

**Jason Wei and Hyung Won Chung from OpenAI** will giving an in-person talk about **Intuitions on Language Models and the Early History and Evolution of Transformers**


Jason: OpenAI, prev GoogleBrain. Popularized ideas like [[Chain of Thought]], [[Instruction-Tuning]], Emergent Phenomenon

Q: Why do LLMs work so well?
- Tool: Manually inspect your data!
	- In 2019, they were trying to build one of the first lung cancer classifiers (what type of Cancer?) He talked to pathologists about how to do it, read many papers, etc. Ultimately got some mileage out of actually learning a lot about the data.

Dartmouth students like to {MASK}
- The language model outputs a probability distribution over every token in the vocabulary

![[Pasted image 20240411163826.png]]
Our actual-word is going to be something like a one-hot vector, in this distribution sense -- so our loss is really going to be the difference between our most likely guess and 1.

Intuition: ==Next Word Prediction = Massively Multi-Task Learning== 🗺️📝🌍🙂

| Task               | Example sentence in pre-training that would teach the task                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Grammar            | In my free time I like to (code, banana)<br>- The LM learns that code is higher probability than banana                |
| Lexical Semantics  | I sent to the store to buy Papapaya, Dragon Fruit, and (Durian, Squirrel)                                              |
| World Knowledge    | The Capital of Azerbaijan is (Baku, London)                                                                            |
| Sentiment Analysis | I was engaged by the movie the whole time, I thought it was (good, bad)                                                |
| Translation        | The word for pretty in Spanish is (bonita, hola)                                                                       |
| Spatial Reasoning  | Iroh went to the kitchen to make tea. Standing next to Iroh, Zuko pondered his destiny. Zuko left the (kitchen, store) |
| Main question      | Arithmetic exam answer key: 3 + 8 + 4 = (15, 11)                                                                       |
The Next-Word-Prediction task is really challenging!


Intuition: ==Scaling compute (data * size of language model) reliably improves loss== 📉
- Pioneered by Kaplan et al 2020
- You can basically predict the loss, given the amount of compute, and it predictably decreases as you increase compute.

|              | Small LM                                                                                     | Large LM                               |
| ------------ | -------------------------------------------------------------------------------------------- | -------------------------------------- |
| Memorization | *Choosey* about what it can and can't memorize                                               | Can memorize *a lot of tail knowledge* |
| Heuristics   | Tend to learn *first-order* heuristics. Trying to do grammar, not necessarily math problems. | Can learn quite complex things.        |

Intuition: While overall loss improves smoothly, ==individual tasks can improve suddenly==! 😱

Overall Loss = $Loss_{grammar} + Loss_{sentiment} + Loss_{knowledge} + ... + Loss_{math}$

But do each of these improve at the same rate as we scale the model? Probably not!
- Some of these get saturated pretty quick as we scale the model (eg grammar).
- Other might not make much progress

![[Pasted image 20240411165050.png]]
It turns out that you can look at a big set of tasks and look at what the shape of the scaling curves are!
He used [[BIG-Bench]], at the 202 tasks -- here's the distribution!


![[Pasted image 20240411165706.png]]
Above:
- inv.scaling = "Inverse Scaling; the accuracy gets worse as you increase the size of the LM!"

![[Pasted image 20240411165710.png]]

The takeaway: ==Plot your scaling curves!==

![[Pasted image 20240411170003.png]]==For research: Plotting the middle dot is SO important in terms of how you see the performance of your new thing improving!== Always consider how your trick interacts with model scale! :)

Q: "What do you think are the current bottlenecks for LLMs?"
A: If you go back to the scaling laws paradigm, what it says is that if you increase the size of the data and model, you get a lot better performance -- and we'll keep pushing on those.

Q: "What are your thoughts on the Mirage paper?"
A: I always get this question! I would encourage you to read the paper and decide for yourself, but what the paper says is that if you change the metric, it looks different; but at the end of the day, I don't think that emergence is a mirage in terms of language model abilities.


-----

Hyung Won Chung: On the OpenAI ChatGPT team. Pretraining, [[Instruction-Tuning]], [[Reinforcement Learning from Human Feedback|RLHF]], previously Google Brain, MIT PhD

Hyung thought about what he should talk about:
- Those on Zoom and in the room should go through and shape the future of AI; let's get it right!

When we talk about something in the future, the best place to get advice is to look at history -- let's look at the early history of the Transformer, and learn lesson from there.

![[Pasted image 20240411170723.png]]

So many things are coming out that it's hard to keep up, regardless of how many years of experience you have. People often spend too much time on the new things, and not enough time on the old things.

Hyung thinks it's important to look into old things!
It's important to study the change itself; how did we get here, from the older thing? 

What does it mean to study the change itself?
1. Identify the ***dominant*** driving forces behind the change
2. Understand the dominant driving forces
3. Predict the future trajectory

![[Pasted image 20240411171101.png]]
![[Pasted image 20240411171153.png]]

How doe this tie to AI?
- New agent, new modality, new MMLU benchmark thing -- I can't even catch up with the latest thing, how do I predict the future of research?
- There's a dominant force governing all of AI research though!

Rich Sutton Plot:
![[Pasted image 20240411171328.png]]

![[Pasted image 20240411171434.png]]
Unfortunate: We model how WE think, incorporate it into a mathematical model, and try to implement it... But do we really understand how humans think? Not really! It seems like a structure that works in the short term, but this sort of behavior really becomes a bottleneck -- we shouldn't limit the degree of freedom that we give to machines!

==The Bitter Lesson==: The past 70 years in AI research can be summed as thus: More general methods with weaker assumptions, add more data and compute. Not fancy little tricks.

![[Pasted image 20240411171642.png]]
Let's choose the less-structure line!


![[Pasted image 20240411171938.png]]
Dominant force: Cheap compute and Scaling
- Let's understand this driving force better!

Let's analyze some key structures made by researchers and see why they might relevant/irrelevant now!


![[Pasted image 20240411172026.png]]

What is a Transformer?
- A sequence model!
- It has as input a sequence
	- Words, images, whatever -- a general concept

First, we tokenize it; we have to represent the words to computers, whcih requires some encoidng scheme; we do it with a fixed number interegers; now we have a sequence of integers.

![[Pasted image 20240411172226.png]]

We then represent them as a sequence of vectors

We want to model the interaction between sequence elements; we take the dot product between them. 
Transformers use [[Attention]] to model this interaction!

Original Transformer was the [[Encoder-Decoder Architecture]]

![[Pasted image 20240411172246.png]]
An example of [[Machine Translation|MT]], which used to be a very cool thing.

We translate an English sentence into German!

Englih sentence -> Tokens -> Embeddings -> Positional Embeddings -> Transformer blocks
- Attention block
	- Bidirectional self attention
	- FFNN
- Repeat this N times

At the end of the encoder block, you have a seqjuence of vectors, each representing the (highly contextualized) word

The Decoder
We put in as input what the answer should be: BOS Das ist gut (in training)

We use Causal Self Attention!
- We can only attend tot he answers that we've already decoded

Simliar transformer blocks

The output includes an EOS token, with no BOS token

---

The [[Cross-Attention]]: Each sequence in the decoder should also attend to the (FINAL) output of the encoder.

## [[Encoder-Only Architecture]]
![[Pasted image 20240411172548.png]]
In this case, the output is just a SINGLE VECTOR that represents the input sequence!

We can then slap a task-specific linear layer/regression/classification head to map to some output

![[Pasted image 20240411172611.png]]
Above: This sort of process was popularized by [[Bidirectional Encoder Representations from Transformers|BERT]] in 2018

## [[Decoder-Only Architecture]]

![[Pasted image 20240411172825.png]]

Common misconception: This is only used for sequence generation in a freeform manner, or dialogue -- you can't use it for MT!
- Nah, you can! Here, self attention can serve both roles.
- We share the same set of parameters here between both input and target sequences 🤔😅


![[Pasted image 20240411172851.png]]
He argues that these are quite similar!
- Let's go from the Encoder-Decoder to the Decoder, and see what some of the differences are, and consider whether these structures are relevant nowadays.

![[Pasted image 20240411172933.png]]
Let's make the left closer to the right...
1. Share cross and self-attention matrices
2. Encoder-decoder uses separate parameters -- can they share parameters?
3. The target-to-input attention pattern; we need to connect the target to the input in the decoder-only!
4. The input attention; We mentioned the bidirectional attention... (???)

![[Pasted image 20240411173201.png]]

![[Pasted image 20240411173212.png]]
Encoder-Decoder has these additional structural biases built in

![[Pasted image 20240411173329.png]]
The first assumption above is useful for MT; at the time in 2017, this was considered a difficult task. You can actually get a [[BLEU]] score pretty fuckin good.
IF the goal is to learn MT, it makes sense to say "This parameter in teh encoder takes care of english, and the other parameters in the decoder take care of german"

But think: Does it make sense for "knowledge in german" and "knowledge in english" to be represented in separate parameters?
- With general, larger models, this assumption feels very unnatural to the speaker

![[Pasted image 20240411173400.png]]
@ Google 2 years ago, they did some IFT work where they finetuned a model on some instruction dataset so that it can understand instructions.

![[Pasted image 20240411173433.png]]
(They spent a lot more time on Palm than they did on T5; why did it improve so much more from IFT?)
![[Pasted image 20240411173454.png]]
Hypothesis: It's about the length
- Academic datasets have distinctive length distribution: Long input and short target
	- This is due to the inherent difficulty of grading long text responses by humans.

He thinks it's possible that having separate params for LONG TEXT in input and SHORT TEXT in output was an effective bias to give the model
- But these days, we have longer generation :O and this might not be suitable?


![[Pasted image 20240411173711.png]]
We know that NNs learn hierarchiacal representations
- So why should the decoder attend only to the final layer of the encoder?

Is bidirectional input attention really necessary?
![[Pasted image 20240411173831.png]]
![[Pasted image 20240411173927.png]]
When we generate "Bad", we have to encode the whole thing!
When we generate "Why," we have to encode the whole thing again!

![[Pasted image 20240411173949.png]]
In contrast, for Unidirectional, we don't have to redo "how" when we generate Bad; the previous stuff can be cached! This makes a big difference when talking about multi-turn conversation.

==Bidirectionality he argues is mostly solved by scale, and we don't need it anymore!==

## Conclusion
![[Pasted image 20240411174033.png]]
