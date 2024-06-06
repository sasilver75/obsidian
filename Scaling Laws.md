January 23, 2020 (5 months before [[GPT-3]])
[[OpenAI]], lead author [[Jared Kaplan]]
Paper: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - "The Scaling Laws Paper"

#zotero 
Significance: Answers the question: What happens as we scale model size, compute, and parameters? Performance smoothly increases. Larger scales are more sample-efficient. Establishes a large number of scaling laws interrelating these variables.

Note: Later revisited by [[Chinchilla]] in March 2022

-----


Notes:
- Seems to consider LMs up to 1B in size
- LM performance improves smoothly as we increase the model size, dataset size, and amount of compute; all must be scaled in proportion, or you'll hit bottlenecks.
- Performance depends strongly on scale, but only weakly (within reasonable limits) on model shape (eg depth, width)
- Large models are more sample-efficient than small models, reaching the same level of performance with fewer data points.
- When working with a fixed compute budget $C$ without any other restrictions on the model size $N$ or the available data $D$ , we obtain optimal performance by training *very large models* and stopping *significantly short of convergence.*
- Maximally compute-efficient training has data requirements growing slowly as $D \sim C^{0.27}$ with training compute.
- Optimal batch size is roughly a power of the loss, and is determinable by measuring the gradient noise scale.
- The paper proposes a variety of scaling laws and relationships, but they're all a little hard to interpret, compared to the Chinchilla paper.
- Given a 10x increase in computational budget, the size of the model should increase 5.5X while the number of training tokens should only increase 1.8x (This is later refuted in the Chinchilla paper).

Abstract
> We study empirical scaling laws for language model performance on the cross-entropy loss. The ==loss scales as a power-law with model size, dataset size, and the amount of compute used for training==, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. ==Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size==. These relationships allow us to ==determine the optimal allocation of a fixed compute budget==. Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence.


![[Pasted image 20240427120636.png]]
This is the figure that some would say motivated the rush of investment into the field; "scale Transformers up, and they continue to get better, with no ending in sight."

![[Pasted image 20240427121428.png]]
Left: Shows that sample efficiency improves as you increase the scale of the model. See that larger models benefit sooner and more from dataset size scaling, and that smaller models plateau earlier (large models don't even plateau in the figure).
Right: Compute-efficient training means stopping short of convergence.

![[Pasted image 20240427123050.png]]

# Non-Paper Figures

![[Pasted image 20240605160210.png]]
From CS685 Lecture 17

![[Pasted image 20240605160423.png]]
From CS685 Lecture 17