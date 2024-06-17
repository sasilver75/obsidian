Link: https://snorkel.ai/llm-distillation-demystified-a-complete-guide/
This article was bad

----

LLMs like ChatGPT, Gemeni, Grok etc. dazzle with performance and breadth, but most applications don't need that breadth -- and you're paying for it in time and latency!

> "Your POS System doesn't need to be able to write haikus!"

Most AI problems don't require the flexibility and generalization that comes with large models -- this is where [[Distillation]] comes in, which enables us to train small, highly-capable models with the help of large models.

LLM Distillation poses a larger generative models as a "teacher," and a smaller model as a "student." 
- The student model might be a simple model like a logistic regression model, or even a 7B-param language model!

In the most basic version of distillation, data scientists might start with unlabeled data and ask an LLM to label it, and then use this synthetically labeled data to train the "student" model.
- ((I call this "Data-Distillation"))

Why would we use LLM distillation?
- Cost: Smaller models cost less to do inference than larger ones, and inference costs dominate training costs.
- Speed: Smaller models do inference faster than larger ones.
- Infrastructure headaches: Hosting private versions of large LMs means wrangling and coordinating significant resources.

What are the drawbacks of LLM distillation?
- The student is limited by the teacher; Generalized LLMs faced with specialized tasks will typically fall short of production-grade accuracy!
- You still need a lot of unlabeled data! The LLM will create labels for you, but source data may be in short supply for any number of reasons.
- You may be limited in what LLMs you can use; some LLMs bar users from using their LLMs output to train potentially competitive models.

Generative LLM distillation with prompts and responses works similarity to LLM distillation for classification models; the main difference is just that data scientists extract responses from the teacher models instead of labels.

## Knowledge Distillation
- Knowledge distillation focuses on training the student models' probability distribution to mimic those of the teacher model; this differs from the approach discussed above, which only care about the teacher LM's output *label*.
- Training on the "soft target" of the teacher model's probability distribution is a richer training signal than just training on the label, but it generally requires that you have both the student and teacher LM running in parallel, which isn't required of data distillation.

Notes the ==MiniLLM== paper as an interesting method of improving knowledge distillation; uses soft targets extracted from the teacher as a target, but ==encourages the student to only focus on high-probability outcomes, rather than on mimicking the entire probability distribution.==

## LLMs distilling themselves; Context Distillation
- [[Context Distillation]]: Seems to refer to using highly-engineered prompts to help the teacher model generate responses... and then stripped the prompt of its engineering and reduce the response to *only* its final answer, to create a dataset that's used to then fine-tune a model.
 ![[Pasted image 20240616235224.png]]

## Generative LLMs to predictive Tasks: Distilling step by Step
- Distillation can be bottlenecked by a lack of *unlabeled* data; if we want a model to classify contract clauses into a dozen categories, we might have very few raw examples to train from!
- In the ==Distilling Step-by-Step== paper from Google/Snorkel, we can fine-tune models for classification tasks on as little as one-eighth as much data as traditional fine-would require.
	- It works by asking the teacher model to return not only its answer, but also the *rationale* behind the answer; it then directs the student model to do the same (to yield both a final response and reasoning).
![[Pasted image 20240617000304.png|300]]
((This just seems like the Orca paper, to me, which IIRC is just distillation on CoT from GPT-4.))
















