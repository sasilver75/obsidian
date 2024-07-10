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
- [[BERT|BERT]] is perhaps the most famous of these transformer architectures; here, we produce vector embeddings for each word (or token) similar to word2vec; but embeddings are much richer thanks to much deeper neural networks.
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

BERT and other Pretrained models (distilBERT, RoBERTa, ALBERT) soon arrived on the scene to help with tasks like classification, Q&A, POS-tagging, and more.

Before [[Sentence Transformers]], the approach to calculating accurate sentence similarity with BERT was to use a Cross-Encoder structure, where we pass two sentences to BERT simultaneously (getting rich interaction between tokens in each), and adding a classification head to the top of BERT, then outputting a similarity score.

![[Pasted image 20240614235204.png|300]]

[[Cross-Encoder]]s produce very good similarity scores, but they aren't *scalable*; if we want to perform similarity search through a small 100K sentence dataset, we need to do 100k forward passes through a large BERT-like network for every query; this could take hours!
- If we wanted to *cluster* sentences in a 100k dataset, that would result in 500M+ forward passes -- not realistic!

Ideally, we'd be able to ==precompute== representations (at least for the documents), so that at query time all we need to do is compute the representation of our query before calculating (eg) the cosine similarity between that query representation and the precomputed document representations.

With BERT, we can build a sentence embedding by either:
- Averaging the values across all token embeddings output by BERT (if we input 512 tokens, then we output 512 embeddings)
- Use the output of first first \[CLS\] token (a BERT-specific token whose output embedding is often used in classification tasks).

Using one of these two approaches gives our sentence embeddings that can be stored and compared much faster, shifting search times from ~65 hours to ~5 seconds! But it's not as accurate as cross-encoders 🤔

The solution was designed by [[Nils Reimers]] and others in [[Sentence-BERT|sBERT]] and the [[Sentence Transformers]] library.
- For 10k sentences, sBERT can produce our embeddings in ~5 seconds and compare them with cosine similarity in ~0.01 seconds!

### Sentence Transformers
- [[Sentence-BERT|sBERT]] is similar to a BERT Cross-Encoder, but drops the final classification head and processes one sentence at a time; it then uses [[Average Pooling|Mean Pooling]] on the final output layer to produce a sentence embedding.
- Unlike BERT, sBERT is fine-tuned on sentence pairs using a *siamese architecture* ((This is just two-towers/bi-encoder stuff, I think)). We can think of this as having two identical BERTs in parallel that share the same network weights.
![[Pasted image 20240615000013.png]]
In reality, we use a single BERT model, but because we process sentence A followed by sentence B as *pairs* during training, it's easier to THINK about this as two models with tied weights.

The siamese BERT outputs two sentence embeddings; the next step is to concatenate these embeddings `u` and `v` (along with $|u-v|$):
![[Pasted image 20240615000333.png|300]]
|u-v| is calculated as the element-wise difference between the two vectors.
- These are all then fed into a FFNN that has *three outputs!*
	- These align to our [[Natural Language Inference|NLI]] similarity labels of 0, 1, 2.

![[Pasted image 20240615000646.png|300]]
When we optimize the model weights, they're pushed in a direction that allows the model to output more similar vectors where we see an *entailment* label, and more dissimilar vectors where we see a *contradiction* label.

==The fact that this training approach works is not particularly intuitive, and has been described by [[Nils Reimers]] as *coincidentally* producing good sentence embeddings!==

Since this paper, further work has been done in the area (eg [[RoBERTa]]) -- we'll be exploring some of these in future articles.

Let's talk about using Sentence Transformers, though:

```python
from sentence_transformers import SentenceTransformers

model = SentenceTransformer('bert-base-nli-mean-tokens')

sentences = [  # 5 of these
	"the fifty mannequin heads floating in the pool kind of freaked them out",
    "she swore she just saw her sushi move",
    ...
]

embeddings = model.encode(sentences)

embeddings.shape  # (5, 768)  768-dim'd embeddings

import numpy as np
from sentence_transformers.util import cos_sim

sim = np.zeros((len(sentences), len(sentences)))

for i in range(len(sentences)):
sim[i:,i] = cos_sim(embeddings[i], embeddings[i:])

sim
```
![[Pasted image 20240615001222.png]]
This shows cosine similarity between every combination of our five sentence embeddings.

Though we got good results from sBERT above, there are other sentence transformer models in the `sentence-transformers` library -- these newer models significantly outperform the original SBERT (which isn't even an available model on the SBERT.net page anymore!)
![[Pasted image 20240615001338.png|300]]
```python
from sentence_transformers import SentenceTransformer

mpnet = SentenceTransformer('all-mpnet-base-v2')
```

## Chapter 3: Training Sentence Transformers with Softmax Loss
- This article dives deeper into the training of the *first* sentence transformer, sBERT; we'll explore the NLI training approach of softmax loss to finetune models for producing sentence embeddings.
	- Recall that this method only "coincidentally works" by Reimer's admission, and has since been superseded by other methods like ==MSE margin== and ==multiple negatives ranking loss==.

We'll cover two approaches for fine-tuning:
1. How NLI training with softmax actually works
2. Uses the excellent training utilities of the `sentence-transformers` library

### NLI Training
- One of the most popular ways to train sentence transformers is using [[Natural Language Inference|NLI]] datasets. We'll use Stanford NLI (SNLI) and Multi-Genre NLI (MNLI) datasets.
	- 943k sentence pairs, where all pairs include a (premise, hypothesis, label)
	- Labels
		- 0 - Entailment
		- 1 - Neutral
		- 2 - Contradiction
- When training the model, we feed sentence A (premise) into BERT, followed by sentence B (hypothesis) on the next step.

```python
import datasets
# Import/split the snli and mnli datasets (omit)
dataset = datasets.concatenate_datasets([snli,m_nli])

# We must convert our human-readable sentences into transformer-readable tokens, so we go ahead and tokenize our sentences.
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for part in ['premise, 'hypothesis']:
	dataset = dataset.map(
		lambda x: tokenizer(
			x[part], max_length=128, padding='max_length', truncation=True
		), batched=True
	)
	for col in ['input_ids', 'attention_mask']:
        dataset = dataset.rename_column(
            col, part+'_'+col
        )
        all_cols.append(part+'_'+col)

# Nowe we prepare the data to read into the model, converting dataset features into PyTorch tensors and initializing a data loader which will feed data into our model during training
 dataset.set_format(type='torch', columns=all_cols)

# Create teh dataloader
batch_size = 16
loader = torch.utils.data.DataLoader(
	dataset, batch_size=batch_size, shuffle=True
)
```

Optimizing with Softmax Loss was the primary method used by Reimer and Gurevych in the original sBERT paper (this is no longer the go-to training approach; instead, the MNR (Multiple Negatives Ranking) loss approach is most common)

```python
model = BertModel.from_pretrained('bert-base-uncased')

# We'll use a siamese BERT architecture during training, and we'll convert the output 512 768-dim embeddings into a single, average embedding using mean-pooling.
def mean_pool(token_embeds, attention_mask):
	# reshape attention mask to 768dim embeddings
	in_mask = attention_mask.unsqueeze(-1).expand(tokekn_embeds.size()).float()
	# perform mean-pooling but exclude padding tokens (specified by in_mask)
	pool = torch.sum(token_embeds * in_mask,1) / torch.clamp(in_mask.sum(1), min=1e-9)

	return pool
```

Here we take BERT's token embedding output and the sentences attention_mask tensor; we resize the attention mask to align to the higher 768-dim of the token embeddings. We apply this resized in_mask to those token embeddings to exclude padding tokens from the mean pooling, and then our mean pooling takes the average activations of values across each dimension to produce a single result.

Next, we concatenate these embeddings. We find that (u,v,|u-v|) is the best, in this paper.

```python
uv_abs = torch.abs(torch.sub(u,v))
x = torch.cat([u,v,uv_abs], dim=-1)
```
Now we feed this into a FFNN, which processes the vector and outputs three activation values, one for each of our label classes (entailment, neutral, contradiction).

```python
ffnn = torch.nn.Linear(768*3, 3)
# ... later ...
x = ffnn(x)
```

With these activations calculated (and our labels known), we can now calculate the softmax loss between them!
![[Pasted image 20240615005655.png|300]]
Softmax loss is calculated by applying a softmax function across our three activation values, producing a predicted label.
- We then use a [[Cross-Entropy]] loss to calculate the difference between our predicted label and the true label.

```python
loss_func = torch.nn.CrossEntropyLoss()

x = loss_func(x, label)
```

The model is then optimized using this loss, with an [[Adam]] optimizer @ 2e-5 LR and a linear warmup of 10% of the total training data for the optimization function.

```python
from transformer.optimization import get_linear_schedule_with_warmup

# Initialize everything first
optim = torch.optim.Adam(model.parameters(), lr=2e-5)

# Setup a warmup for the first ~10% steps
total_steps = int(len(dataset) / batch_size)
warmup_steps = int(.1*total_steps)
scheduler = get_linear_schedule_with_warmup(
	optim, num_warmup_steps, warmup_steps,
	num_training_steps=total_steps - warmup_steps
)

# Later, during the training loop we update the scheduler per step
scheduler.step()
```

Now, all togehter!
```python
from tqdm.auto import tqdm

# 1 epoch should be enough, increase if wanted
for epoch in range(1):
    model.train()  # make sure model is in training mode
    # initialize the dataloader loop with tqdm (tqdm == progress bar)
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # zero all gradients on each new step
        optim.zero_grad()
        # prepare batches and more all to the active device
        inputs_ids_a = batch['premise_input_ids'].to(device)
        inputs_ids_b = batch['hypothesis_input_ids'].to(device)
        attention_a = batch['premise_attention_mask'].to(device)
        attention_b = batch['hypothesis_attention_mask'].to(device)
        label = batch['label'].to(device)
        # extract token embeddings from BERT
        u = model(
            inputs_ids_a, attention_mask=attention_a
        )[0]  # all token embeddings A
        v = model(
            inputs_ids_b, attention_mask=attention_b
        )[0]  # all token embeddings B
        # get the mean pooled vectors
        u = mean_pool(u, attention_a)
        v = mean_pool(v, attention_b)
        # build the |u-v| tensor
        uv = torch.sub(u, v)
        uv_abs = torch.abs(uv)
        # concatenate u, v, |u-v|
        x = torch.cat([u, v, uv_abs], dim=-1)
        # process concatenated tensor through FFNN
        x = ffnn(x)
        # calculate the 'softmax-loss' between predicted and true label
        loss = loss_func(x, label)
        # using loss, calculate gradients and then optimize
        loss.backward()
        optim.step()
        # update learning rate scheduler
        scheduler.step()
        # update the TDQM progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

```
Above: Here, we're only training for a single epoch here. Realistically, this should be enough (and mirrors the SBERT paper). The last thing we do is save the model:

```python
import os

model_path = './sbert_test_a'

if not os.path.exists(model_path):
	os.mkdir(model_path)

model.save_pretrained(model_path)
```

### Fine-Tuning with Sentence Transformers
- The `sentence-transformer` library has excellent support for those of us who want to train a model without worrying about the underlying training mechanisms.

...omitted...

## Chapter 4: Training Sentence Transformers with Multiple Negatives Ranking Loss
(Skipping to the interesting parts about MNR)
- MNR and Softmax approaches both use a siamese-BERT architecture during fine-tuning. For each step, we process a sentence A (anchor) into BERT, followed by sentence B (our positive)

We can extend this further with triplet-networks. In the case of triplet networks for MNR, we would pass three sentences, an anchor, its positive, and its negative.
- We're NOT using triplet networks here, so we removed the negative rows from our dataset.

BERT outputs 512 768-dimensional embeddings. We convert these to averaged sentence embeddings using mean-pooling. Using a siamese approach, we'll produce two of these per step -- one for our anchor, and another for our positive.

We then calculate the cosine similarity between each anchor and ALL of the positive embeddings in the same batch!

From here, we produce a vector of cosine similarity scores (of size batch_size) for each anchor embedding a_i. It's assumed that each anchor should share the highest score with its positive pair, p_i.

To optimize for this, we use a set of increasing label values to mark where the highest score should be for a_i, and categorical cross-entropy loss. ((?))

...

## Chapter 5: Multilingual Sentence Transformers

## Chapter 6: Unsupervised Training for Sentence Transformers

## Chapter 7: An introduction to Open Domain Question-Answering

## Chapter 8: Retrievers for Question-Answering

## Chapter 9: Readers for Question-Answering

## Chapter 10: Data Augmentation with BERT

## Chapter 11: Domain Transfer with BERT

## Chapter 12: Unsupervised Training with Query Generation (GenQ)

## Chapter 13: Generative Pseudo-Labeling (GPL)
