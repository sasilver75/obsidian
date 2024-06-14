https://www.pinecone.io/learn/series/nlp/
A mini-course

----
Two pillars support semantic search; vector search and NLP. In this course, we focus on the pillar of NLP and how it brings "semantic" to semantic search. We introduces concepts and theory before backing them up with real, industry-standard code and libraries.


## Chapter 1: Dense Vectors
- There's perhaps no greater contributor to the success of modern NLP than [[Word2Vec]] in 2013; it's one of the most iconic and early examples of dense vectors representing text.

Dense vs Sparse Vectors
- Sparse vectors can be stored more effectively and allow us to perform syntax(word)-based comparisons of two sequences.
	- Sparse vectors are called sparse because the vectors are sparsely populated with information -- typically many zeroes and a few ones.
- While sparse vectors represent text syntax, we can view dense vectors as numerical representations of semantic meaning.
	- Dense vectors can still be highly dimensional (eg 784-dimensions), but each dimensions contains relevant information.

![[Pasted image 20240614115908.png|300]]

If we create dense vectors for every word in a book, then reduce the dimensionality of those vectors to visualize them in 3D, we'll be able to identify clusters and relationships between data (man/woman/king/queen example).

- Many techniques exist for building dense vectors, whether we're representing words, sentences, MLB players, or even images.
- We'll explore:
	- 2vec methods
	- Sentence Transformers
	- Dense Passage Retrievers (DPR)
	- Vision Transformers (ViT)

### Word2Vec
- Word2Vec wasn't the first, but it was the first widely used embedding model thanks to being very good and coming with the word2vec toolkit, which allows easy training or use or pretrained word2vec embeddings.
![[Pasted image 20240614120225.png|300]]
- In Word2Vec, given a sentence, we create a word embedding by taking a specific word (one-hot encoded) and mapping it to surrounding words through an encoder-decoder ((autoencoder?)) neural net.
	- In the [[Skip-Gram]] version of Word2Vec, given a word, we predict the surround words (its context). The representation at the bottleneck is said to be the embedding for the word.
	- In the [[Continuous Bag of Words]] (CBOW) method, we aim to predict a word based on its context.
- Both methods are alike in that they produce a dense embedding vector from the middle hidden layer of the encoder-decoder network.

To compare longer chunks of text effectively, it'd be convenient if they were represented by a single vector!
- Several *extended* embedding models like ==sentence2vec== and ==doc2vec== cropped up.

Since Word2vec, some superior technologies for building dense vectors have entered the scene, so we don't often see "2vec" methods in use today.

### Sentence Similarity
- Transformer models produce incredibly information-rich dense representations which can be used for a variety of downstream tasks.
- [[Bidirectional Encoder Representations from Transformers|BERT]] is perhaps the most famous of these transformer architectures; here, we produce vector embeddings for each word (or token) similar to word2vec; but embeddings are much richer thanks to much deeper neural networks.
- The [[Attention]] mechanism allows BERT to prioritize which context words should have the biggest impact on a specific embedding by considering the alignment of said context words.
	- But we want to focus on comparing *sentences,* not words -- and BERT embeddings are produced for each token, which doesn't help us directly in comparison of sentences.
	- The first transformer build explicitly for this was [[Sentence-BERT|sBERT]] (Sentence Bert), a modified version of BERT -- but sBERT has a context limit of only 128 tokens, being unable to create single vector embeddings for anything beyond this limit. Many of the latest models allow for longer sequence lengths!

Let's look at using the [[Sentence Transformers]] library to create some sentence embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

sentences = [
    "it caught him off guard that space smelled of seared steak",
    "she could not decide between painting her teeth or brushing her nails",
    "he thought there'd be sufficient time is he hid his watch",
    "the bees decided to have a mutiny against their queen",
    "the sign said there was road work ahead so she decided to speed up",
    "on a scale of one to ten, what's your favorite flavor of color?",
    "flying stinging insects rebelled in opposition to the matriarch"
]

embeddings = model.encode(sentences)
# We can create some 768-dimensioned embeddings
embeddings.shape  # (7, 768) 
```

Even though our most semantically-similar sentences about bees and their queen share *zero* descriptive words, our model correctly embeds them closely in vector space, when measured with cosine similarity.

### Question Answering
- Another widespread use of transformer models is for [[Question Answering]]; within QA, [[Open-Domain]] QA is an interesting task -- it allows us to take a big set of sentences/paragraphs that contain answers to our questions. We then ask a question to return a small chunk of one or more paragraphs which best answer our question.

We usually have three components:
- Some sort of database to store our sentences/passages/contexts.
- A ==retriever== that retrieves contexts that it sees as similar to our question.
- A ==reader model== which extracts the answer from our related contexts.

![[Pasted image 20240614121931.png|300]]

The retriever portion of this architecture is our focus here.
- We want a model that can map the following question-answer pairs to the same point in vector space:
	- "What is the capital of France" (query)
	- "The capital of France is Paris" (answer)

One of the more popular models for this ((Unsure year)) is Facebook's [[Dense Passage Retrieval]] (DPR) model.
Consists of two smaller models:
- A context encoder
- A query encoder
Each of these uses the BERT architecture, and they're trained in parallel on question-answer pairs; we use a [[Contrastive Loss]] function (the difference between the vectors output by each encoder). Our goal is to match query embeddings and context embeddings to the same space so that they can be easily retrieved (eg using MIPS).
![[Pasted image 20240614122209.png|300]]

Let's look at using the HuggingFace `transformers` library to play with DPR:

```python
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

# Load our Passage Encoder+Tokenizer from weights on the hub
ctx_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Load our Query Encoder+Tokenizer from weights on the hub
question_model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
```

Now, given a question and several contexts, we can tokenize and encode like so:
```python
questions = [
    "what is the capital city of australia?",
    ...
    "how many searches are performed on Google?"
]

contexts = [
    "canberra is the capital city of australia",
    ...
    "Google serves more than 2 trillion queries annually",
]

# For our questions and contexts, we can tokenize them and then encode them using the appropriate tokenizer/encoders we created earlier
xb_tokens = ctx_tokenizer(contexts, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
xb = ctx_model(**xb_tokens)

xq_tokens = question_tokenizer(questions, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
xq = question_model(**xq_tokens)

xb.pooler_output.shape  # torch.Size([9, 768])
xq.pooler_output.shape  # torch.Size([3, 768])
```

Now that we've got query and passage embeddings, we can compare our query embeddings `xq` to all of our context embeddings `xb` to see which are the most similar with [[Cosine Similarity]].

```python
import torch
for i, xq_vec in enumerate(xq.pooler_output):
	probs = cos_sim(xq_vec, xb.pooler_output)
	argmax = torch.argmax(probs)
	print(questions[i])
    print(contexts[argmax])
    print('---')

what is the capital city of australia?
canberra is the capital city of australia
---
what is the best selling sci-fi book?
the best-selling sci-fi book is dune
---
how many searches are performed on Google?
how many searches are performed on Google?
---

```

Cool! So it seems like we returned correct answers for two out of our three questions. It's clear that DPR isn't the *perfect model*, given our dataset.
- On the positive side, in ODQA we often return *many contexts*, and allow a reader model to identify the best answers. Reader models can "re-rank" contexts, so retrieving the top context immediately isn't required in order to return the right answer.

### Vision Transformers (ViT)
- Computer vision has become the stage for exciting advances from transformer models, which were historically restricted to NLP until [[Vision Transformer|ViT]]s came along.
- We can encode images and texts into the same vector space, with interesting results! ViTs have ben used in OpenAI's [[CLIP]] model, which uses two encoders (a ViT as our image-encoder, and a model like BERT for text). Thee models are trained in parallel on image-caption pairs and optimized using a contrastive loss function, producing high-similarity vector for image-text pairs in the large training dataset.

![[Pasted image 20240614123618.png|300]]

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

captions = [
	"a dog hiding behind a tree",
	"two dogs running",
	...
]

# Image are PIL images I think
inputs = processor(
	text=captions, images=images, return_tensors='pt', padding=True
)

outputs = model(**inputs)

probs = outputs.logits_per_image.argmax(dim=1)
probs  # tensor([2, 0 ,1])

for idx, image in enumerage(images):
	argmax = probs[i].item()
	print(captions[argmax])
	plt.show(plt.imshow(np.asarray(image)))

# (Prints out image/caption pairs, with the captions seeming to describe the images pretty well.)
```

Nice, now we have flawless image-to-text matching using CLIP!


## Chapter 2: Sentence Transformer and Embeddings
- Before diving into sentence transformers, it might be nice to look back at why transformer embeddings are so much richer than (eg) word2vec.
![[Pasted image 20240614131733.png|300]]
This information bottleneck between two models means that we're creating a massive amount of information over multiple timesteps and trying to squeeze it *all* through a single connection. This limits performance, because much of the information produced by the decoder is lost before reaching the decoder.
- The [[Attention]] mechanism provided a solution to the bottlnekc issue, offering another route for information to pass through.
![[Pasted image 20240614131911.png|300]]
Now the information bottleneck is removed, and there's better information retention (of the most relevant information) across longe sequences.

The Transformer (2017) encoder-decoder model removed the need for RNNS through the use of three key components:
1. Positional Encoding
2. Self-Attention
3. Multi-head Attention

BERT and other Pretrained models (distilBERT, RoBERTa, ALBERT)



## Chapter 3: Training Sentence Transformers with Softmax Loss

## Chapter 4: Training Sentence Transformers with Multiple Negatives Ranking Loss

## Chapter 5: Multilingual Sentence Transformers

## Chapter 6: Unsupervised Training for Sentence Transformers

## Chapter 7: An introduction to Open Domain Question-Answering

## Chapter 8: Retrievers for Question-Answering

## Chapter 9: Readers for Question-Answering

## Chapter 10: Data Augmentation with BERT

## Chapter 11: Domain Transfer with BERT

## Chapter 12: Unsupervised Training with Query Generation (GenQ)

## Chapter 13: Generative Pseudo-Labeling (GPL)
