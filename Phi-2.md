December 12, 2023
Paper: [Phi-2: The surprising power of small language models](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)

Language Model Phi-2, at ==2.7B parameters==, is twice the size of the 1.3B Phi-1 and Phi-2 models.
Trained on "text-book quality" synthetic data. Seems to punch above its weight, like the rest of the Phi family.

"Abstract" (It's a blog post, not a paper)
> Over the past few months, our Machine Learning Foundations team at Microsoft Research has released a suite of small language models (SLMs) called “Phi” that achieve remarkable performance on a variety of benchmarks. Our first model, the 1.3 billion parameter [**Phi-1**(opens in new tab)](https://huggingface.co/microsoft/phi-1), achieved state-of-the-art performance on Python coding among existing SLMs (specifically on the HumanEval and MBPP benchmarks). We then extended our focus to common sense reasoning and language understanding and created a new 1.3 billion parameter model named [**Phi-1.5**(opens in new tab)](https://huggingface.co/microsoft/phi-1_5), with performance comparable to models 5x larger.
> We are now releasing [**Phi-2**(opens in new tab)](https://ai.azure.com/explore/models/microsoft-phi-2/version/4/registry/azureml-msr), a ==2.7 billion-parameter language model== that demonstrates outstanding reasoning and language understanding capabilities, ==showcasing state-of-the-art performance among base language models with less than 13 billion parameters==. On complex benchmarks ==Phi-2 matches or outperforms models up to 25x larger==, thanks to new innovations in model scaling and training data curation.
> With its compact size, Phi-2 is an ==ideal playground for researchers==, including for exploration around mechanistic interpretability, safety improvements, or fine-tuning experimentation on a variety of tasks. We have made [**Phi-2**(opens in new tab)](https://ai.azure.com/explore/models/microsoft-phi-2/version/4/registry/azureml-msr) available in the Azure AI Studio model catalog to foster research and development on language models.

Adulation for this model was slightly tempered by allegations of (perhaps unintentional) benchmark hacking/leaking. "With only 2.7B parameters, Phi-2 matches the performance of Mistral and Llama-2 models at 7B and 13B parameters on various aggregated benchmarks."

Shows that training data plays a critical role in model performance; Phi-2's training data mixtures contains synthetic datasets specifically created to teach the model common-sense reasoning and general knowledge, including science, daily activities, theory of mind, among others.

![[Pasted image 20240424120912.png]]