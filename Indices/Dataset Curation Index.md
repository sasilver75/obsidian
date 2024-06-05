Resources:
- [BLOG: HuggingFace RefinedWeb Creation](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)

1. It's common to train a model on a given corpus considered "clean" (eg Wikipedia) and use it to check the perplexity on the dataset that we're trying to curate.
	- This doesn't always correlate with improved performance on downstream tasks of interest.
2. Another method is to train small (cheap, fast) models on a representative subset of our dataset, and evaluate them on a set of evaluation tasks set of evaluation tasks.
	- It's important to choose diverse and representative evaluation tasks, and try not to overfit to any individual benchmark.
3. Another way to compare different datasets is to train a model on each dataset and have humans rate/compare the generations of the models, eg on [[ChatBotArena]]. 
	- This might provide the most reliable results, but it's expensive and slow, and simple pretrained (not-instruction-tuned) models aren't yet prepared to be assistants and might be very sensitive to prompt details.




