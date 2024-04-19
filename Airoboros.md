Github Link: [airoboros](https://github.com/jondurbin/airoboros)
From [[Jon {Durbin]], a "Customizable implementation of the [[Self-Instruct]] paper."
Referenced by [[Nous Research]] folks in the [[Capybara]] series of models as being a useful tool for creating datasets.

> Jon Durbin: This is *my take* on implementing the Self-Instruct paper. The approach is quite heavily modified, and ==does not use any human-generated seeds==.

> Problem and proposed solution:
> - Models can only ever be as good as the data they are trained on.
> - High quality data is difficult to curate manually, so ideally the process can be automated by AI/LLMs.
> - Large models (gpt-4, etc.) are pricey to build/run and out of reach for individuals/small-medium business, and are subject to RLHF bias, censorship, and changes without notice.
> - Smaller models (llama-2-70b, etc.) can reach somewhat comparable performance in specific tasks to much larger models when trained on high quality data.
> - The airoboros tool ==allows building datasets that are focused on specific tasks, which can then be used to build a plethora of individual expert models. This means we can crowdsource building experts==.
> - Using either a classifier model, or simply calculating vector embeddings for each item in the dataset and using faiss index/cosine similarity/etc. search, incoming requests can be routed to a particular expert (e.g. dynamically loading LoRAs) to get extremely high quality responses.

