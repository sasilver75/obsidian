---
aliases:
  - Knowledge Distillation
  - Distilled
  - Distill
---
Traditional knowledge distillation involves transferring knowledge from a larger, more complex model (teacher) to a smaller, simpler model (student). Generally this involving training on the logits of the teacher, rather than on the data labels, but it can also refer to creating a set of labeled data using a teacher model that a student model can later train on.

Variant: [[Self-Distillation]]


Note:
- I've heard "Distillation" used in two contexts:
	- "Model Distillation" where a weaker model is trained on the probability distributions (a strong training signal) of the larger model
	- "Dataset Distillation" in the context of synthetic data, where we use (eg) GPT-4 to *distill* a set of synthetic data (eg using a technique like that from [[Self-Instruct]])


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
