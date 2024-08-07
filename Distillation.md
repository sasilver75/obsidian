---
aliases:
  - Knowledge Distillation
  - Distilled
  - Distill
---
Traditional knowledge distillation involves transferring knowledge from a larger, more complex model (teacher) to a smaller, simpler model (student). Generally this involving training on the logits of the teacher (model distillation), rather than on the data labels, but it can also refer to creating a set of labeled data using a teacher model that a student model can later train on (data distillation).

Variant: [[Self-Distillation]]
Variant: ==[[Online Distillation]]== (eg Gemeni Flash being distilled from 1.5 Pro *as Pro trained!*) (relevant [paper](https://arxiv.org/abs/1804.03235))

==Soft-Target Distillation== (logits, model distillation) vs ==Hard-Target Distillation== (label, data distillation)

==Off Policy Distillation== (usual) vs ==On-Policy Distillation== (See [[Gemma 2]])


Note that dataset contamination is something to think about when you're doing hard-target distillation (eg) from some model that itself have been contaminated.

Note:
- I've heard "Distillation" used in two contexts:
	- "Dataset Distillation"/Hard target distillation in the context of synthetic data, where we use (eg) GPT-4 to *distill* a set of synthetic data (eg using a technique like that from [[Self-Instruct]]), and then train a model on that dataset. We're basically training a student to predict the *labels* predicted by the teacher.
	- "Model Distillation"/"Knowledge Distillation"/Soft target distillation where a weaker model is trained on the probability distributions (a strong training signal) of the larger model.
		- The MiniLLM paper modifies this slightly, and rather than encouraging the student model to mimic the entire probability distribution, it encourages the student to only focus on high-probability outcomes.
	- There's also what I'll call "Representation Distillation," where the student is guided to have similar representations to the teacher model at intermediate layers of the model.



Possible Distillation objectives, from least-to-most heavy duty (weighted averages of elements of this list are common):
0. Distill student by training on Gold data for the task
1. Train the student to have the same outputs as the teacher.
	- This doesn't actually require that you have the teacher at distillation time, just a dataset of labeled results.
2. Train the student to have the same *output scores* as the teacher
	- The centerpiece of one of the most famous distillation papers, *Hinton et al, 2015*
	- Requires the teacher at distillation time, because we require those score vectors.
	- ((This usually refers to the softmax output of the teacher model; the *probabilities*))
3. Train the student to have the same final *output states*
	- Requires much more access to the teacher at distillation time
	- From the [[DistilBERT]] paper
	- ((This usually refers to the raw logits from the final layer of the teacher model, before the softmax function is applied; more fine-grained information about the model's decision-making process. The cosine loss suggests that the student model is trained to match the *direction* of these logit vectors, rather than the exact values)).
4. Train the student to have similar hidden states and embeddings as the teacher
	- With an intuition that the student will be more powerful and alike the teacher if it has similar internal representations
	- Requires full access to teacher @ distillation time
1. Train the student to mimic the counterfactual behavior of the teacher under interventions -- instances where we change the internal state of the teacher, and do the same corresponding thing to the student, and make sure they have corresponding behavior.
	- This is relatively new, Chris Potts involved 
	- Requires full access to teacher @ distillation time


Modes of distillation:
1. Standard distillation: Teacher has frozen parameters, only the student parameters are updated
2. Multi-teacher distillation: Simultaneously try to distill various teachers into a single student that can perhaps perform multiple tasks
3. Co-distillation: Student and teacher and trained jointly. Also called "online distillation" (Anil et al, 2018)
4. Self-distillation: The objective includes terms that seek to make some model components align with others from the same model (Zhang et al. 2019)


![[Pasted image 20240617110315.png|300]]

An interesting subject is the limitations of many organizations when it comes to using data generated by closed-source models with restrictive usage, eg GPT-4.
- It's an interesting fact that many of the “*you can’t train on outputs from our model*” clauses in licenses actually come from the data providers used in training the model rather than the model owners themselves. They need these clauses to protect their business due to the strength of synthetic data.

![[Pasted image 20240723132706.png]]
An overview of knowledge distillation from [[Sebastian Raschka]] in the context of computer vision

One hallmark of Gemma 2 is its relatively large vocabulary size (256,000 tokens); twice the size of the LLaMA 3 vocabulary, and 8 times the size of the Phi-3 vocabulary (32,000).
- A large vocabulary size allows for better coverage of words and concepts, improved handling of multilingual content, and reduce tokenization artifacts, but it also comes with tradeoffs like increased model size and potentially slower inference due to larger embedding and output layers.
	- ((This is contrary to my understanding about the effect of vocabulary size))

There's also an interesting section on "logit capping", a technique he hadn't seen before
- It's a form of min-max normalizing and clipping logit values to keep them in a certain range; authors presumes this is to improve stability and gradient flow during training.

Authors also leverage *model merging* techniques to combine multiple models from different runs that use different hyperparameters, though the paper doesn't provide much detail about this.

In terms of modeling and performance, Gemma 2 is almost as good as the 3x larger LLaMA 3 70B.