#article 
LInk: https://www.interconnects.ai/p/llm-synthetic-data

----

> Synthetic data is the accelerator of the next phase of AI; what it is, and what it means.

The notion of synthetic data (data created by a human rather than a machine) has a long history in NLP and ML broadly -- it's usually closely tied to the notion of Data Augmentation, where a piece of data can be modified slightly to add diversity to the dataset.

One of the older links in NLP is [[Back-Translation]], where synthetic data is a *new translation task* from the mode's outputs to the origianl text.

---
Aside: [[Back-Translation]]
- Given an input text in some *source language* (eg English)
- Translate this text to some temporary *destination* language (eg French)
- Translate *back* the previously translated text into the source language (eg English)
---

Today, synthetic data has taken on a much grander task -- removing humans from the loop of making AI both aligned and enjoyable to use -- a task spearheaded by Anthropic's training methods and OpenAI's mysterious new Superalignment team, tasked with using AI feedback to solve alignment (because humans won't be powerful enough).

In the meantime, synthetic data has become a go-to resource for many of the most popular boutique, open model providers fine-tuning Meta/Mistral's models.
- There are even rumors that Mistral is just Llama 2 pretraining continued on GPT 4 tokens!

# Can Synthetic Data provide the next breakthrough?
- The current/next generation of models will have likely trained on all the high-quality data on the internet, with the latest sources coming from things like YouTube and podcasts. 
- Model providers will be looking for new directions to get the last few orders of magnitude of data needed for scaling laws to hold. A core assumption behind proponents of synthetic data at scale is that simply adding more data will make the model better at solving long-tail tasks/evaluations.

Nato thinks that we have ~2 more generations of models for scaling to be worth it, as computational costs skyrocket.

==The argument against synthetic data follows that all the data that we're generating is from the same distribution as the current best models, so some do not expect the SOTA to be advanced by it.==

