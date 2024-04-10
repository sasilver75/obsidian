#article 
Link: https://cameronrwolfe.substack.com/p/dolma-olmo-and-the-future-of-open?utm_source=post-email-title&publication_id=1092659&post_id=141461162&utm_campaign=email-post-title&isFreemail=true&r=764e6&utm_medium=email

-------
![[Pasted image 20240220210052.png]]

The increasing popularity of open-source LLMs is widely considered to be beneficial, but there's one problem: ==the definition of open-source varies drastically between models.== 
- In many cases, open-source LLMs simply ==release their model parameter and inference code, but OBFUSCATE important details like the contents of the pretraining dataset, training hyperparameters, evaluation strategy, training logs, checkpoints, and more.==

Here, we will overview the recently-released [[Dolma]] dataset and [[OLMo]] suite of LLMs, from [[Allen Institute|AI2]]

The process of constructing the Dolma pretraining corpus, released under the AI2 impact licenses, is fully-documented -- authors even release code and tooling for re-building the dataset from scratch!

Similarly, the model itself is released under an Apache-2.0 license along with all training data, code, checkpoints, training logs, and more.

Dolma and OLMo take massive steps toward demystifying the LLM preparing process by releasing all relevant resources and analyzing any and all choices made in the process of training OLMo.


# Dolma 🍇🍃 : An Open Corpus of 3T Tokens for LLM Pretraining Research

> The most powerful LMs are built by a few organizations who withhold most model development details; The composition of LM pretraining data is often only vaguely stated, even in cases where the model itself is released for public use.

![[Pasted image 20240220211043.png]]
Above: The Dolma corpus
- [[CommonCrawl]]: A free, open repository of web crawl data, with over 250B pages... 3-5B pages added per month.
- [[The Stack]]: A 3TB dataset of permissively licensed source code in 30 programming languages from [[Eleuther]].
- Reddit: You know what it is-
- [[C4]]: A *Colossal, Cleaned Common Crawl* -- it's basically a subset of the good parts of Common Crawl.
- PeS2o: A collection of 40M creative open-access academic papers, derived from the Semantic Scholar Open Research Corpus, or S2ORC -- these are both from AI2. 
- [[Project Gutenberg]]: An online library of free eBooks.

The ==process of constructing this composite dataset, including design principles and the empirical reasoning behind each decision, is documented in the paper==.
- Additionally, the authors even ==released a full toolkit for efficiently reproducing and modifying resulting dataset!==

Dolma design principles
- The end goal of Dolma is to:
	1. Create the largest open (data and process) pretraining corpus
	2. Use this pretraining corpus to train an open language model (weights, code, checkmarks, process, logs), called [[OLMo]]

Authors aimed to pull from data sources that are commonly used in prior work, and to process the data that aligns with existing best practices. In areas where best practices are unknown, empirical practices are adopted, whereby smaller variants of OLMo are trained and evaluated to determine the downstream impact of changes to the pretraining dataset.

The impact of data size on model performance was investigated in [[Chinchilla]], where authors showed that larger models must be pretrained over sufficiently large pretraining datasets to truly optimize their performance.
- But even smaller models like LLaMA-2 models haven't yet fully converged after pretraining over 2T tokens!
- As a result, authors aimed to make Dolma sufficiently large (ie 3T tokens) to facilitate further studies on the pretraining dataset size.


# Prior Open Pretraining Datasets
- Dolma isn't the only effort that has tried to shed light on the process of constructing pretraining data for LLMs.
- [[C4]]: Colossal Clean Crawled Corpus
	- This pretraining dataset was originally constructed for the [[T5]] model; ==This data is sourced from Common Crawl==, and the full dataset contains about 750GB of text in total. Authors chose to create the C4 dataset from scratch due to the lack of high-quality, publicly-available pretraining datasets for language models.
	- ==Many models (Gopher, Chinchilla, MPT, and more) use C4 as a subset of their pretraining data, since the quality is quite high.==
- [[ROOTS]]: Responsible, Open-Science Open-Collaboration Text Sources
	- Developed as part of the BigScience research initiative for training [[BLOOM]]
	- A ==multilingual dataset== that spans over 46 natural languages and 13 programming languages. The ==English portion of this dataset is limited in scale -- too small for training English-only LLMs!==
	- Comprised of over 1.6TB of text in total.
- [[RefinedWeb]]
	- The pretraining corpus that was curated to train the [[Falcon]] LLMs.
	- Corpus is ==comprised of web data that undergoes a purely-heuristic-based filtering pipeline== (ie no *model-based* filtering), and ends up with a high-quality pretraining dataset. 
	- Large 5T token corpus, but ==only a small portion of the data (600B) was made available to the public.==
- [[RedPajama]]
	- An initiative led by [[TogetherAI]] to produce a leading open-source LLM and release it to the public with a more permissive license.
	- As a first step, the authors recreated the pretraining dataset for LLaMA resulting in the RedPajama v1 and v2 datsets
	- Similar to Dolma in nature, but RedPajama has a more limited scope; ==the creators of RedPajama are trying to reproduce LLaMA, whereas Dolma attempts ot provide a transparent resource that empowers others to study all notable aspects of LLM pretraining==, and pulls from a variety of additional sources (eg scientific papers, code, conversational forums, more).


# Creating the Dolma Corpus
- ==The process of constructing a pretraining dataset for an LLM consists of three primary components:==
	1. ==Acquisition==: obtaining content from a variety of sources
	2. ==Cleanup==: Using heuristics and model-based techniques to filter the data
	3. ==Mixing==: Deduplication and up/down-sampling of data sources

![[Pasted image 20240220233435.png]]
Above: Some of the data cleaning pipelines applied to various types of data in the Dolma corpus.

To process data for pretraining, we rely on ==four primary transformations of our data==:
1. ==Langauge filtering==
2. ==Quality filtering==
3. ==Content filtering==
4. ==Deduplication==

------
#### Aside: What is fastText?
- ==fastText== is a free and open-source library that can be used to learn text representations and train lightweight text classifiers.
- fastText classifiers have simple architectures that take an average of word representations in a sentence, and then pass the result to a linear classifier. They perform well -- similar to deep-learning based classifiers, but are cheaper. They can be improved my moving to an n-gram representation of our data, rather than the default unigram word representations.
- fastText is used in the cleaning pipelines in Dolma, and in other large-scale data processing applications.

Note: The ==CCNet== data pipeline is a popular reference architecture for creating high-quality, deduplicated, and monolingual text datasets from web data. It uses a fastText classifier for language identification.

-------

1. ==Language Filtering==
	- The Dolma dataset is ==*English-only!*==
	- To accomplish this, we need to make a classifier tool for filtering out non-English content. We use a lightweight classifier -- we use the ==CCNet pipeline with a fastText language identification model== to predict the primary language of each document -- and then only keep the documents with scores above a certain threshold.
	- It's true that a certain amount of non-English data will always be present -- these classifiers aren't perfect.
	- ![[Pasted image 20240221140546.png]]
2. ==Quality Filtering==
	- Refers to the process of ==removing "low quality" text from a pretraining corpus==.
	- What's the definition of "low quality" text? Researchers quibble about it, and how to filter it out. Should we use solely heuristics (e.g. Gopher, Falcon), or should we use machine learning models should be used for quality filtering (eg LLaMA)?
	- ==In Dolma, the authors avoid model-based filtering techniques and rely solely upon heuristic filtering methods.==
	- The authors ==use different heuristics for different data sources== -- for example, filtering conversations based on length, community votes, and flags from moderators, in the context of conversational data.
3. ==Content Filtering==
	- Refers to ==removing harmful text -- primarily toxic content and personally identifiable information==.
	- Again, ==to identify toxic content==, we train a pair of fastText classifiers, which are then used to tag (and remove) spans of toxic text, to classify hateful and NSFW content based on the Jigsaw Toxic Comments dataset.
	- The authors adopted a conservative threshold for removing toxic text -- a text span must be classified as toxic with a relatively high probability for it to be removed... in an effort to avoid removing *too much* data from the corpus.
	- ==To detect PII==, a series of regular expressions are adopted to find spans of text corresponding to email addresses, IP addresses, and phone number. ==These bits of text are either masked, or the entire document is removed from the corpus (if more than 5 pieces of PII are detected).==
	- ![[Pasted image 20240221140930.png]]
4. ==Deduplication==
	- Recent research has shown that deduplicating an LLM's pretraining dataset makes the model's training more data-efficient.
	- Crafting a robust (and efficient) deduplication pipeline is an incredibly important aspect of building pretraining corpora for LLMs.
	- Authors perform three stages of deduplication for web text:
		1. ==URL-based==: Eliminates web pages scraped multiple times
		2. ==Document-based==: Eliminates pages that contain exactly the same text
		3. ==Paragraph-based==: Eliminates individual paragraphs with the same text.
	- All stages above are implemented efficiently by using a ==Bloom Filter==.
	- For other sources of data beyond web data, a similar strategy is used for deduplication, though paragraph-level deduplication may be unnecessary for sources with shorter documents.

### Putting it all together
- When constructing Dolma, authors perform preprocessing steps in a *specific order* to ensure efficiency!
	- Namely, ==URL and document-level deduplication are performed first, followed by quality and content filtering, while paragraph-level deduplication is performed last==.


# OLMo: Accelerating the Science of Language Models 📈
- [[OLMo]] is a set of truly open LLMs that are trained on [[Dolma]].
- OLMo models match the performance of SoTA open LLMs, and can be used to more-deeply study hte science of language modeling.
- The OLMo suite of models are completely open, and come with a variety of artifacts:
	- Model weights
	- Training code
	- Evaluation code
	- Adaptation code
	- Training logs
- ==OLMO is comprised of five models in total: *four 7B models and a 1B model*==
- Prior "open" LLMs aren't often truly open, since they don't release their data (+ dataset construction process), training/evaluation code, model weights, inference code, and more... And they don't even always have a highly-permissive license.
	- [[Mistral]] provides model weights and a brief report
	- [[LLaMA 2]] provides detailed alignment instructions, but *minimal information about the pretraining dataset.*
	- [[MPT]] provides detailed instructions about constructing the model's pretraining dataset, but *doesn't actually release that data*.
		- ((I think there was some kerfuffle about this -- [[Jonathan Frankle]] talked on a podcast about not wanting to release the dataset after some backlash, because the dataset contained likely copywritten material from authors, and he wasn't sure that he wanted to promote that))
	- [[Falcon]] releases a *partial subset* of the model's pretraining data, along with a report on the model and data.
	- Both [[Pythia]] and [[BLOOM]] release training code, model checkpoints, and training data, but their *license is restrictive*.


# The Evaluation Process
- OLMo models are evaluated using two different techniques: *==perplexity== and ==downstream task evaluation==*

### What is Perplexity?
- Prior to understanding the perplexity evaluation, we should probably understand the concept of perplexity in general... Put simply, perplexity is a metric that can be used to evaluate a language model by measuring how well it predicts known samples of text.
![[Pasted image 20240221142658.png]]
- When autoregressively generating output via next-token-prediction, the LLM predicts a probability distribution over potential next tokens. From these token-level probabilities, we can easily compute the probability for a sentence generated by the LLM via the product rule of probabilities:
![[Pasted image 20240221142846.png]]
Above: ((It's just, for each token, multiplying the built-up probability by the probability of generating the next token, given the previous tokens. Sort of like flipping two heads being .5 * .5, except we use conditional probabilities, since the probability of a given token is conditioned on the previously generated tokens))

The metric above can give us a good idea of how well a language model "fits" the data.
- The model should assign high probabilities to valid, high-quality sequences of text.
- The ==problem with this probability is that it's highly sensitive to the length of the sentence== -- long sequences will multiply more sub-1 probabilities, making a smaller number!
	- To solve this, we take a geometric mean to normalize the probability computed above by the length of the sentence:
![[Pasted image 20240221143129.png]]
- Above: We normalize the textual sequence probability using a *geometric mean*.
	- ((Unlike the arithmetic mean (which you calculate by adding up all the numbers and dividing by the count of the numbers), the geometric mean is calculated by multiplying all the numbers together and then taking the �nth root of the product (where �n is the number of numbers in the set).))
	- ((Both are measures of central tendencies, but arithmetic mean should be used when the numbers being averaged have vaguely uniform in size, whereas geometric means are better when the numbers follow somewhat of an exponential distribution. In our case with perplexity, we're multiplying a bunch of sub-1 probabilities together... so the product of these tends to decrease exponentially with the addition of more terms. Using an arithmetic mean here wouldn't make sense because it can't accurately represent these central tendency of these probabilities.))

*How does this relate to perplexity?*
- ==[[Perplexity]] is the reciprocal of the geometric mean of this sequence probability!==
	- ==Low perplexity== values indicate that a language model assigns *high* probability to textual sequences used for evaluation, and therefore fits the evaluation data *well*.
	- ==High perplexity== values indicate that a language model assigns *low* probability to textual sequences used for evaluation, and therefore fits the evaluation set *poorly*.

#### Perplexity Evaluation
- ==To perform perplexity-based evaluation, authors construct an evaluation dataset called Perplexity Analysis for Language Model Assessment ([[Paloma]]) by aggregating textual sequences from a diverse set of 585 domains collected across 18 different sources of textual data==, and evaluate the LLM by measuring perplexity on textual sequences from this dataset.
- Compared to prior work, Paloma significantly improves the diversity of perplexity-based evaluation benchmarks, allowing us to determine whether an LLM can accurately model text across a wide variety of domains.
	- Paloma significantly improves the *diversity* of perplexity-based evaluation benchmarks, allowing us to determine whether an LLM can accurately model text across a variety of domains.

#### Downstream task evaluation
- While perplexity-based evaluations are useful for understanding whether an LLM understands a doamin of text well (eg measuring perplexity over a corpus of scientific publications to determine if the LLM captures this data), but ==perplexity-based evaluations fail to directly measure how well an LLM performs on downstream tasks!==
- For this, authors evaluate the model using the [[Catwalk]] framework, which provides a standardized abstraction for evaluating various LLMs across a wide variety of tasks and datasets. ==Models are solely evaluated using a zero-shot prompting strategy!==
- ![[Pasted image 20240221151710.png]]
- For OLMo, they selected nine reasoning tasks that were similar to the task set used to evaluate LLaMA and LLaMA 2.

> "We perform downstream evaluations to make decisions around model architecture, initialization, optimizers, learning rate schedule, and data mixtures. We call this our *==online evaluation==*, as it runs in-loop every 1,000 training steps, and provides an early and continuous signal on the quality of the model being trained."

- In-loop evaluations:
	- Beyond the offline evaluation of OLMo described above, we see that OLMo undergoes similar evaluations in an online fashion... Namely, researchers test a variety of different model hyperparameters and rely upon online evaluations performed every 1,000 training steps to evaluate these choices. These evaluations include both the perplexity and downstream task metrics described above.


## Model Architecture
- The OLMo LLM is based on the [[Decoder-Only Architecture]] of Transformers. 
	- Put simply, the standard [[Transformer]] has two components: an *encoder* and a *decoder*. The decoder-only architecture only uses the decoder component of the transformer. 

The OLMo suite:
- Three model sizes of OLMo are included in the release
	- The 1B and 7B parameter models are released along with the writeup, but at the time of writing, the 65B model is still training and will be released in the near future.
- The architecture choices are quite simple to other peer LLMs -- OLMo shares the same hidden dimension, number of heads/layers, and MLP ratio as LLaMA 2... but OLMo does have a slightly shorter context window -- only 2K tokens -- compared to LLaMA 2.

![[Pasted image 20240221152256.png]]

SwiGLU Activations
- Each transformer block passes intermediate activation values (ie the output of feed-forward and attention layers) through an activation function.
- The [[Rectified Linear Unit|ReLU]] activation function is standard in most deep NN architectures -- LLMs tend to adopt a different suite of (more complex) activation functions -- OLMo adopts the [[SwiGLU]] activation function, which was used in recent LLMs like [[PaLM]] and [[LLaMA 2]].

![[Pasted image 20240221152437.png]]

The Swish activation function is a smoother function compared to ReLU and has been shown to achieve better performance in several applications.
- SwiGLU is a combination of this Swish activation with a GLU activation.
- SwiGLU requires *three* matrix multiplications, which makes it more compute-intensive compared to activation functions like ReLU -- but it improves LLM performance to use SwiGLU, it seems.

Non-parametric [[Layer Normalization]]
- Each block of the transformer architecture contains intermediate layer normalization operations, as formulated in the figure below.
- ![[Pasted image 20240221152658.png]]
- We normalize a value using the mean and variance of all values in each input sequence. A small additive constant is included in the denominator alongside the variance to avoid issues with taking a square root of zero, or dividing by zero.
- After layernorm is applied, we typically have two learnable parameters that apply an element-wise affine transformation to the module's output.

Better positional embeddings
- Instead of using absolute positional embeddings as used by the original transformer architecture, more modern LLMs (including OLMo) choose to adopt a [[Rotary Positional Embedding]] (RoPE) strategy.
	- [[Rotary Positional Embedding|RoPE]] ==combines the benefits of absolute and relative positional embeddings by==:
		1. Encoding the absolute position of each token with a rotation matrix
		2. Directly injecting relative position information into the self-attention operation
	- Using this approach, both the absolute and relative position of each token can be captured, letting the LLM generalize to longer input sequence lengths.

The Tokenizer
- The creators of OLMo use the GPT-NeoX tokenizer, which is found to be well-suited to Dolma due to the fact that it was trained over a corpus of web data (C4 dataset), and has a permissive license -- which isn't true of all tokenizers! For example, using LLaMA 2's tokenizer would cause the license from LLaMA-2 to apply to OLMo! 😱

Other design choices
- All bias terms are excluded from the OLMo architecture, which is shown to improve training stability
- Authors adopt sequential (as opposed to parallel) transformer block formulation.
- No weight tying is used by OLMo.
- The model uses a vanilla (full) variant of [[Multi-Headed Attention]] (self) -- in contrast, several recent models have adopted [[Grouped Query Attention]] and [[Multi-Query Attention]] , which are variants that improve the inference/decoding efficiency of attention, perhaps at the cost of reduced performance, by sharing key and value vectors across attention heads. The OLMo authors forego these due to their impact on performance.
![[Pasted image 20240221154903.png]]


## The Training Process
- Trained over 2T token subset of Dolma, but the model may be trained for more than one epoch over this data.
	- For example, OLMo-1B is trained on 2T tokens (one epoch) while OLMo-7B is trained over ~2.5T tokens (~1.25 epochs).
		- The authors claim that repeating data during training doesn't negatively impact model performance.
- All models are trained using the ZeRO optimizer strategy and PyTorch's FSDP framework.
- All OLMo models are trained using the [[AdamW]] optimizer.
- Training is replicated on clusters of both NVIDIA and AMD GPUs to ensure portability of OLMo (resulting models on each cluster perform nearly identically, but required slight changes in hyperparameters).

# Empirical Analysis of Dolma and OLMo
- We can study questions as:
	- Does decontamination of pretraining data impact model performance?
	- Does learning from code make LLMs better reasoners?
	- How does pretraining data mixture impact the LLM's knowledge base?

#### The Impact of Pretraining Data on Performance
- Authors trained several OLMo-1B models -- these models are used to compare strategies for data decontamination and mixing in particular, and evaluated using the benchmarks discussed previously (perplexity via Paloma, Catwalk for downstream tasks).

Decontamination
- The probability of testing and evaluation data being "leaked" within any given model's large pretraining datasets is quite high -- ==large-scale language corpora often contain copies of benchmarks used for evaluating LLMs! So one might argue that the impressive performance of LLMs on downstream tasks could be attributed to test set leakage== -- maybe our models just memorized the answers to these tasks during pretraining!
- The impact of data contamination is a hotly debated topic.
- To evaluate the impact of data contamination, we train OLMo-1B over subsets of RedPajama, and test how it performs on contaminated and decontaminated versions of dataset. The decontamination approach is shown to not have a clear negative (or positive) impact on model's performance.

Data Mixology
- After pretraining data is curated across several sources, we need to decide ==how to actually "mix" the data, and how to upsample or downsample each source of data to create the final dataset.==
- The mixing strategy is an important hyperparameter that has a massive impact on the LLM's performance, but data mixology for LLMs is largely a black box due to a lack of rigorous analysis in AI literature.
- The authors of OLMo try several different mixing strategies, and ==the high-level takeaway is the simple fact that the chosen data mixture has a noticeable impact on the model's ability to capture certain subdomains.==
	- As such, one should avoid domain mismatches between the pretraining dataset and inference-time incoming data of LLMs ((this seems intuitive, but is nice to know.))

Evaluating the OLMo suite
![[Pasted image 20240221160758.png]]
Above: See that OLMo is about as good as MPT, LLaMA, Falcon, LLaMA2, etc. It's importantly markedly better than Pythia, which is sort of "replaces" as a truly "open" model.

# Conclusion
- Dolma and OLMo take a massive step towards improving LLM pretraining!
- Complete openness of each will help researchers, and associated tools (training/eval cocde, data toolkit) allow researchers to test new ideas and variations of the approach used.































