
Demystifying Mixtral of Experts
Albert Jiang, works at [[Mistral]] + PhD Student @ Cambridge

Context
- Architecture
	- Dense transformers
	- Sparse MoE (sMoEs)
- Interpreting SMoEs
	- Mostly talking about the *Mixtral of Experts* paper than came out in January, 2024


----

# Context: Mistral 7B

Mistral 7B: A dense transformer
- [[Grouped Query Attention]]
	- Somewhat between multi-headed attention and mulit-query attention
	- ![[Pasted image 20240425163302.png|300x200]]
- [[Sliding Window Attention]]
	- For each token position, we attend to a relatively short span of previous tokens; as we go later in the slayers, we have indirect access to even distant tokens.
	- ![[Pasted image 20240425163358.png|300x200]]

These are the standard configurations for Mistral 7B
![[Pasted image 20240425163440.png|300]]

He's using "Noam Shazeer Naming Convention" in the following example of a Transformer Layer

{Missed picture, fuck}
- First, initialize your Q,K,V matrices; notice that we don't use a bias term in any of these matrices.
- We also write down out output matrix; we know that it has the same dimensions as the transpose of the query matrix.
- To write an attention_forward layer, we assume that our input x has dimensions LxD
	- We use our matrices to find our query, key, and value matrices 
	- We apply out rotary embeddings to query and key matrices (nto values)
	- We do a standard attention mechanism with our queries, keys, and values
		- Note: We repeat the keys and values a couple times to make the dimension match with teh queries.
	- Return the output
- To have a transformer layer, we also then have nomralization (We use RMSNorm)
- And the n do an attention residual, norm and dense, then dense residual


It performed well!
Let's get to Mixture of Experts now

----

# Mixture of Experts
- Papers
	- [[Switch Transformer]]: Scaling Trillion Parameter Models with Simple and Efficient Sparsity (Google Paper, 2022 - [[Noam Shazeer]], Barret Zoph, William Fedus)
	- Outrageously Large Neural Networks: The Sparsely Gated Mixture of Experts Layer (Google, 2017; at the time, we were still using RNNs. We had a gating network deciding which experts to route to.)

In our implementation of the MoE layer, we do something similar:
![[Pasted image 20240425163900.png]]
Inputs are sent to router; router decides gating weights, we pick the top-two experts. After those experts process inputs, we take a router-score-weighted combination of those two expert outputs.

![[Pasted image 20240425164004.png]]
Each token doesn't have to go through the entire NN's parameters; it can just go through the active parameters for that token. The Mixtral of Experts token can outperform Llama 2 70B with ~5x faster inference!

## What does MoE-ifying MLP layers actually give you?
*Conventional* explanation for MoE:
- While Attention layers implement algorithms for reasoning, MLP layers store knowledge.
- Thus, by MoE-ifying MLPs, we're supposed to get a boost in knowledge.

If MoE-ifying MLP can give a boost in knowledge, what about MoE-ifying attention layers?
- The Switch Transformer player addressed this
	- Replaced the trainable Q, K, V matrices with switch layers; You have a gating layer, then some dense layers.
	- Problems with stability at the time; bf16 can diverge, but fp32 doesn't 🤷‍♂️
	- Are there better stability techniques? Research needed!


### Myth 1: There are 8 experts in Mixtral 8x7B
- No, *EVERY LAYER HAS 8 EXPERTS!*
- The gating network weights to give to the experts, and chooses the top-2. 
- So we actually have 32 x 8 = 256 experts in total, in the network, and they're relatively independent across layers.


### Myth 2: There are 56B parameters in Mixtral 8x7B
- The gating and attention layers are shared
- There are 56.7B parameters in total


### Myth 3: Cost-active parameters
- Mixtral 8x7B has fewer active parameters than LLaMA2 13B
- Having expert routing means that we have more communication cost, though
- While you gain much more in performance/cost, the absolute cost is not proportional to active parameter count. Usually, your MoE... if you have active parameters... your actual cost of serving this models is a little more than the equivalent dense model with the same number of "active" parameters.


##### Research Question: How do we balance loads at inference time?
- Your gating layer might still decide to do unbalanced loading at inference time, which makes inference slower. 
- Some nice ideas exist around Mixture of Depths and dynamic loading based on scores.

##### Research question: How do we compress smoes?

![[Pasted image 20240425164827.png]]
![[Pasted image 20240425164841.png]]
MoE compression is different from normal model compression.


# Interpretation of MoEs
- Deep NNs are hard to interpret, since weights and activations live in high-dimensional spaces
- Attention offers SOME interpretation opportunity, but quickly gets messy as we get more attention heads.
- The gating layer in SMoEs tells you which experts are looking at which tokens --- can we then make sense out of this?
- 
- 

## Myth 4: You want experts to specialize in domains
- It doesn't happen in reality; it seems they specialize almost at token levels


# Treasure hunt
- Someone tried removing the i'th expert from all of the layers of Mixtral, and saw:
![[Pasted image 20240425165821.png|300]]
It seems, if you remove the third expert, that the accuracy just collapses -- intriguing!
![[Pasted image 20240425165851.png|300]]
Meme: Expert 3 is doing all the work

Research question: How do we interpret MoE decisions? What are the features that they're learning?
- Experts might capture features that are very different concepts than what WE se as concepts!
- Might be more efficient to represent linear combinations of concepts ... ((?))


Conclusion
- Sparse MoE models leverage sparsity to gain more knowledge
- You can train very good MoEs to be efficient at inference
- Expert specialization isn't as straightforward as you might think!




















