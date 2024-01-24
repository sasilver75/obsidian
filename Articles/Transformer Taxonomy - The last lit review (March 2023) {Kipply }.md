---
tags:
  - article
---
Link: https://kipp.ly/transformer-taxonomy/

----------

This document is a running literature review for people trying to catch up on AI; it covers:
- 22 models
- 11 architectural changes
- 7 post-pre-training techniques
- 3 training techniques
- 5 grab-bag items

Everything's loosely in the order of importance and somewhat uniqueness.
Systems/performance and alignment are excluded, to be saved for another article.


# (1/5) Models

### [[GPT-3]] (May 2020)
175B params, 96 layers, 12288 embedding dimension, 96 heads
- ==A seminal paper for LLMs==, following both the [[GPT-2]] paper and the [[Scaling Laws]] paper. Trained on a 300B token dataset of mostly Common Crawl, along with some books, webtext, and wikipedia.
- BPE tokenizer
- Alternates dense and sparse attention layers
- Warms up to .6x10^-4 learning rate in first 375M tokens, cosine-decayed to 10% after 260B tokens.
- Batch size ramps from 32k to 3.2M tokens over the first 12B tokens.
- 4x MLP projection ratio as done in the 2017 transformer paper.
- 50k vocabulary size.
- ==Many of these characteristics form a standard recipe that has been reused by later models.==
### [[GPT-4]] (March 2023)
- ==A model of unknown architecture== ("transformer-like").
- The technical report contains mostly evals, as well as results of their continued scaling, which are accurately extrapolated from smaller models as per Scaling Laws.
- The report also documents safety mitigation, and has a demo of their multi-model capabilities, which seem trained á la [[Flamingo]]. 
- It also has ==the best Acknowledgement section of all time==, which reads like a movie credits.
	- ((This is true, it's fucking sick))
### [[Gopher]] (December 2021)
- DeepMind's first LLM release in D

### [[AlphaCode]]

### [[RETRO]]

### [[GPT-3.5]]

### [[Chinchilla]]

### [[Flamingo]]

### [[Gato]]

### Anthropic LM

### [[PaLM]]

### GPT-NeoX

### [[GPT-J]]

### [[GLaM]]

### LaMDA

### Switch

### [[BLOOM]]

### Galactica

### [[LLaMa]]

### Jurassic J1-Grande v2

### [[OPT]]

### GLM-130B

# (2/5) Architectural Changes
- 

# (3/5) Post-Pre-Training Techniques
- 


# (4/5) Training Techniques
- 


# (5/5) Grab Bag
- 




















