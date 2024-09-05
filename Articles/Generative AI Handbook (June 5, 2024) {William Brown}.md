https://genai-handbook.github.io/

This is basically a compendium of a bunch of great resources, with some content to string it together. It's supposed to be "The guide this person wish they had."

Authors is William Brown, CS PhD @ Columba + AI/ML Research @ MorganStanley

I realized about halfway through that this blog didn't really say anything interesting, it moreso just had a bunch of links. I harvested the ones that look interesting that I didn't already read.

---
Some of the resources that he quotes are the following, and it's nice to see that none of them are surprising to me.

- Blogs:
    - [Hugging Face](https://huggingface.co/blog) blog posts
    - [Chip Huyen](https://huyenchip.com/blog/)’s blog
    - [Lilian Weng](https://lilianweng.github.io/)’s blog
    - [Tim Dettmers](https://timdettmers.com/)’ blog
    - [Towards Data Science](https://towardsdatascience.com/)
    - [Andrej Karpathy](https://karpathy.github.io/)’s blog
    - Sebastian Raschka’s [“Ahead of AI”](https://magazine.sebastianraschka.com/) blog
- YouTube:
    - Andrej Karpathy’s [“Zero to Hero”](https://karpathy.ai/zero-to-hero.html) videos
    - [3Blue1Brown](https://www.youtube.com/c/3blue1brown) videos
    - Mutual Information
    - StatQuest
- Textbooks
    - The [d2l.ai](http://d2l.ai/) interactive textbook
    - The [Deep Learning](https://www.deeplearningbook.org/) textbook
- Web courses:
    - Maxime Labonne’s [LLM Course](https://github.com/mlabonne/llm-course)
    - Microsoft’s [“Generative AI for Beginners”](https://microsoft.github.io/generative-ai-for-beginners/#/)
    - Fast.AI’s [“Practical Deep Learning for Coders”](https://course.fast.ai/)
- Assorted university lecture notes
- Original research papers (sparingly)

# Chapter 1: Preliminaries
- Calculus and LA are pretty much unavoidable if you want to understand modern deep learning, which is largely driven by MatMuls and backpropagation of gradients.
- Computers will do a lot for us, but it's still important to have a working knowledge of concepts like:
	1. Gradients and their relations to local minima/maxima
	2. The chain rule for differentiation
	3. Matrices are linear transformations for vectors
	4. Notions of basis/rank/span/independence/etc.

If you math is rusty, watch 3b1b's Essence Of Calculus and Essence of Linear Algebra series!
- The [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) book shows how these topics lay the foundations of optimization problems faced in ML.

(Some Python)

Organization: This document is divided into several sections and chapters:
1. Foundations of Sequential Prediction
2. Neural Sequential Prediction
3. Foundations for Modern Language Modeling
4. Finetuning Methods for LLMs
5. LLM Evaluation and Applications
6. Preference Optimizations for Efficient Inference
7. Sub-Quadratic Context Scaling
8. Generative Modeling beyond Sequences
9. Multimodal Modals

----

# Section 1: Foundations of Sequential Prediction


### Statistical Prediction and Supervised Learning

For probability, it helps to understand:
- Random variables, expectations, and variance
- Supervised vs unsupervised learning
- Regression vs classification
- Linear models and regularization
- Empirical risk minimization
- Hypothesis classes and bias-variance tradeoffs

For general probability theory, understanding how the CLT works is a reasonable litmus test for how much you'll need to know about random variables before tackling some of the later topics we'll cover.
Linear Regression

### Time Series 

Time Series Analysis: How much do you need to know about it in order to understand the more complex generative AI methods?
- Just a tiny bit for LLMs, and good bit more for diffusion

For modern Transformer-based LLMs, it's useful to know about
- The basic setup for sequential prediction problems
- The notion of an autoregressive model

When we get to state space models, a working knowledge of linear time-invariant systems and control theory will be helpful for intuition, but diffusion is really where it's most essential to dive deeper into stochastic differential equations to get the full picture

### Online Learning and Regret Minimzation

It's debatable as to how important it is to have a strong grasp on regret minimization, but a basic familiarity is useful:
- Points arrive one at a time in arbitrary order
- We want low average error across this sequence

Most algorithms designed for this looks basically like gradient descent. The most direct connection to it is when we look at GANs in Section VIII.
- Practical gradient-based optimization algorithms have their roots in this field as well, following the introduction to Adagrad.

He suggests skimming the first chapter of this "[Introduction to Online Convex Optimization](https://arxiv.org/pdf/1909.05207)" book by Elad Hazan to get a feel for the goal of regret minimization

### Reinforcement Learning
- Comes up most directly when finetuning models in Section IV, and may also be useful for thinking about "agent" applications and some of the "control theory" notations that come up for state-space models.
- Be comfortable with the basic problem setup for Markov decision processes, notion of policies and trajectories, and high-level understanding of standard iterative + gradient-based optimization methods for RL.

He recommends Lilian Weng's blog post as a dense starting point ([A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)) and the textbook from Sutton and Barto.

Also this series "[reinforcement Learning by the Book](https://www.youtube.com/playlist?list=PLzvYlJMoZ02Dxtwe-MmH4nOB5jYlMGBjr)" from Mutual Information is a great companion to the Sutton and Barto book, along with some deep RL, using 3b1b style visualizations.


### Markov Models
- Running a fixed policy in a Markov Decision Process yields a Markov Chain; processes resembling this kind of setup are ptetty abundant, and many branches of ML involve systems under Markov assumptions.
- Added some bookmarks 
- Markov Markov models are also at the heart of many Bayesian methods; PRML has some Bayseian angles on ML topics



# Section 2: Neural Sequential Prediction
- Recommends the excellent "[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY): from scratch, in code, spelled out" video series from Andrej Karpathy
	- Recommends 3b1b Neural Networks
	- Statquest videos with Josh Starmer

#### Embeddings and tokens
-ADded posts about LDA nad TFIDF to get a feel for what numerical similarity or relevance scores can represent in language.
The Illustrated Word2Vec from Jay Alamar and the [CS224n Course Notes](https://web.stanford.edu/class/cs224n/readings/cs224n_winter2023_lecture1_notes_draft.pdf) are really good
Lilian WEng Post

#### Encoders and Decoders
- Recurrent models can be configured to both input and output either a single object or an entire sequence! This observation enables seq2seq encoder-decoder architectures, which rose to prominence in the Transformer paper.
- [Again, CS224n course notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf)
- The 
#### Decoder-Only Transformers
- 3B1B's What is a GPT
- Blogpost from Cameron Wolfe

# Section 3: Foundations for Modern Language Modeling



# Section 4: Finetuning Methods for LLMs



# Section 5: LLM Evaluations and Applications




# Section 6: Performance Optimizations for Efficient Inference



# Section 7: Sub-Quadratic Context Scaling



# Section 8: Generative Modeling beyond Sequences


# Section 9: Multimodality
