
#article 
Link: https://eugeneyan.com/writing/nlp-supervised-learning-survey/

---

It's hard to keep up with the rapid progress of NLP! 
Eugene wrote this to organize his thoughts, compare papers, and sort them chronologically, which helps his/our understanding of how NLP (and its building blocks) evolved over time.

He's writing the summary in broad strokes, seeing how NLP has progressed from 1985 till now:

1. Sequential Models
	- [[Recurrent Neural Networks|RNN]]
	- [[Long Short Term Memory|LSTM]]
	- [[Gated Recurrent Unit|GRU]]
2. Word embeddings
	- [[Word2Vec]]
	- [[GloVe]]
	- FastText
3. Word embeddings *with context*
	- [[ELMo]]
4. Attention
	- Transformer
5. Pre-training
	- [[ULMFiT]]
	- [[GPT]]
6. Combining the above
	- [[Bidirectional Encoder Representations from Transformers|BERT]]
7. Improving BERT
	- [[DistilBERT]]
	- ALBERT
	- [[RoBERTa]]
	- XLNet (2019)
	- Big Bird
	- Multilingual Embeddings
8. Everything is text-to-text
	- [[T5]]


# (1) Sequential models to process a sentence (1985)
- [[Recurrent Neural Networks]] ==differ from feedforward neural nets in that their hidden layers have connections to themselves, allowing them to operate over sequences with *something* like memory (hidden state).==
- The state of the hidden layer at one time-step is used as *input* to (the same) hidden layer at the *next* timestep, which is where the name *recurrent* comes from.
	- This allows the hidden information to learn information about the temporal relationships between the tokens in the sequence.
	- See more in Andrej Karpathy's excellent post on [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- ==RNNs have difficulty with modeling long-range dependencies== (words that are far apart in a long sentence) due to the [[Exploding Gradients]]/[[Vanishing Gradients]] problems.
	- The repeated product of partial derivatives being multiplied together through backpropagation may become very small (when derivatives are < 1) or very large (when derivatives are < 1)

- [[Long Short Term Memory|LSTM]] architecture in 1997 improved on the problems with gradients using *gates*, including:
	- ==Forget Gate==
		- Decides what information from the current input and previous hidden state to *forget*, via a sigmoid
	- ==Input Gate==
		- Decides what information to remember ("store" in hidden state) via a sigmoid
	- ==Output Gate==
		- Decides what the next hidden state should be

- These gates improve how the LSTM learns -- and inform what should be forgotten and what should be remembered.
- Though the LSTM was introduced in 1997, it wasn't until 2015 that they saw commercial use in google Voice, Apple's QuickType, Siri, Alexa, and other products.

- [[Gated Recurrent Unit]] (GUs) simplified the LSTM in 2014. It has only two gates:
	- ==Update gate== (similar to LSTM's forget and input gate)
	- ==Reset gate== (which decides how much to forget)
- The GRU has fewer gates (and thus fewer math operations), and so it's faster to train. Eugene says that GRUs converge faster with greater stability.

# (2) Word embeddings to learn from unlabeled data (2013)
- In 2013, [[Word2Vec]] was introduced!
- Though unsupervised learning, it represents words as numbers, or more precisely, ==dense vectors of numbers== (previously, this was done via one-hot encodings).
- It's able to ==learn on large corpora of unlabelled data== (eg Wikipedia), and ==then be used on a variety of downstream tasks==, like Classification, where ==they greatly improve model performance==.

- There are *two ways to train Word2Vec models!*
	1. ==[[Continuous Bag of Words]]== (CBOW)
		- Predict the *center* target word, given the context words around it.
		- *Trains faster* and has a better representation of *rare words*.
	2. ==[[Skip-Gram]]==
		- Predict the *surrounding* context words, given a center word. (Similar to CBOW, but in reverse!)
		- Found to work better with *smaller* amounts of data, and to represent *rare words* better.
		- {Eugene: "I've usually found Skip-gram to work better."}

![[Pasted image 20240326222110.png|450]]
*Above: CBOW vs Skip-gram versions of training a Word2Vec model*

![[Pasted image 20240326222200.png|450]]

Word2vec applied *==subsampling==*, where words that occurred relatively frequently were dropped out with a certain probability, which accelerated learning and improved word embeddings for rare words.

This also tweaked the problem slightly -- instead of predicting the most probable nearby words (out of all possible words), it tries to predicts whether the word-pairs (from skip-gram) were actual pairs!
- This changes the final layer from a softmax with all the words (expensive) to a sigmoid that does binary classification (much cheaper).

==*Negative sampling*== is done to generate negative samples based on the distribution of the unigrams (to add to the positive sample word pairs that we have).


[[GloVe]] was introduced a year later (2014).
- Whereas w2v learns word co-occurrence via a sliding window (i.e. *local statistics*), ==GloVe learns via a co-occurrence matrix (i.e. *global statistics*)==
- GloVe ==then trains word vectors so their differences predict co-occurrence ratios==.
- Surprisingly, w2v and GloVe hare different starting points, but their word representations turn out to be similar.

There are also several variants of W2V that learn subword or character embeddings
- One approach is FastText

# (3) Improving word embeddings with context
- In traditional word embeddings (e.g. Word2Vec, GloVe), each token has *only one representation*, regardless of how it's used in a sentence!
- For example, for "date":
	- They went out on a "date"
	- What "date" is it today?
	- She is eating her favorite fruit, a "date"
	- The photo looks "dated"
Above: Al of these "dates" have the same embedding vector!

[[ELMo]] (2018) improves on this by ==providing word representations *based on the entire sentence!*==
- It does this via a bi-directional language model (biLM), which in ELMo is comprised of a two-layer bidirectional LSTM.
- By going both "left-to-right" (LTR) and "right-to-left" (RTL), ELMo can learn more about a word's context!
	- These embeddings are learned via separate LMs (LTR LM and RTL LM) and concatenated before being used downstream.
- Pre-trained ELMo can be used in a variety of supervised tasks.
- First, the biLM is trained and the word representation layers are frozen. Then, ==the ELMo word representation (i.e. vector) is concatenated with the "normal" token vector to enhance the word representation in the downstream task (e.g. classification).==

![[Pasted image 20240326230625.png]]


# (4) Attention to remove the need for recurrence (2017)
- Recurrent models (e.g. RNN, LSTM, GRU) have a sequential nature -- each hidden state requires the input of the previous hidden state. ==Thus, training cannot be parallelized==.
- Furthermore, ==they can't learn long-range dependencies well==; while LSTM and GRU improved on the RNN, they too had their limit.
- The [[Transformer]] architecture in 2017 solved both problems with [[Attention]]!
	- Attention ==determines how other tokens in the input sequences should be weighted/considered when encoding the current token.==
	- Together with [[Positional Encoding]] of tokens, ==we can process the entire sequence at once, with no recurrence,== and compute each word's representation based on the entire sequence!

- The Transformer is made up of *encoder* and *decoder* stacks. 
	- In each *==encoder stack==*, there are six identical sub-layers, each having a [[Self-Attention]] mechanism followed by a fully connected feedforward neural network.
	- The *==decoder stack==* is similar, but includes an additional attention layer to learn attention over the *encoder's* input! This is [[Cross-Attention]].

[[Multi-Head Attention]]
- The paper uses eight heads, with each head being randomly initialized.
- The outputs from these eight heads are concatenated and multiplied by an additional weight matrix. 
- ==In the decoder stack, the attention mechanism is masked== (to prevent looking ahead at future tokens).

# (5) Fine-tuned learning embeddings (2017)
- So far, we mostly used word embeddings directly, or concatenated them with input tokens (i.e. ELMo) -- there's no fine-tuning of the word embeddings for specific tasks. This changed with ULMFiT!
- [[ULMFiT]] (2017) uses AWD-LSTM (LSTM with dropout at the various gates) as its language model and introduced a fine-tuning phase as part of these steps.
	- First, in ==general-domain LM pre-training==, the LM is trained on *unlabeled data* (eg Wikipedia) {{Usual language modeling task on generic text}}
	- Then, in ==target-task fine-tuning==, the LM is fine-tuned with the *corpus of the target task* (no *labelled data introduced yet*). {{Usual language modeling task on domain-specific text}}
		- [[Discriminative Learning Rate]]s are used, with each layer being fine-tuned with different learning rates, with the last layer having the highest learning rate, and the earlier layers having progressively reduced learning rates.
	- Finally, ==target-task classifier fine-tuning==, where we add two additional linear blocks on the LM (ReLU + softmax). [[Gradual Unfreezing]] is done, where we start with unfreezing the last LM layer and fine-tuning it; one by one, each subsequent layer is unfrozen and tuned. {{Stick on a classification head and train it, using gradual unfreezing, on domain-specific classification tasks}}

[[GPT]] (2017) also added unsupervised pre-training. It uses the Transformer's *decoder stcack* only -- it has an advantage over LSTMs because it performs better w.r.t. long-range dependencies, and is not recurrent in nature (making it easier/quicker to train in a parallel fashion).
- First, ==unsupervised pre-training== (like ULMFiT's first step) involves learning on a corpus to predict the next word.
- Then, ==supervised fine-tuning== tweaks the decoder block for the target task. Task-specific inputs and labels are passed through the pre-trained decoder block to obtain the input representation (i.e. embedding). This is then fed into an additional linear output layer.


# (6) BERT: No recurrence, bidirectional, pre-trained (2018)
- Towards the end of 2018, [[Bidirectional Encoder Representations from Transformers]] was introduced. It obtained SOTA results on *eleven* NLP tasks, impressing the field! This was possible by using elements from previous models:
	1. Transformer (BERT uses the Transformer encoder stack)
	2. ELMo (BERT has bidirectional context)
	3. ULMFiT and GPT (Also adopts pre-training and fine-tuning)
	4. GPT (Has a unified architecture and a single input representation)

- Input to BERT is represented as a single sentence or a pair of sentences. BERT uses [[WordPiece]] embeddings, and introduced a special classification token `CLS` that is always the first token in a sequence, with a `SEP` token separating sentence pairs.

Pre-training is done via two unsupervised tasks:
1. A masked language model (LM) is trained via the ==cloze== task (==because the standard, predict-the-next-word task can't be used with bidirectional context and multiple layers==).
	- BERT masks 15% of token randomly with `MASK` tokens (and/or a random token)
	- The LM predicts the original token with cross-entropy loss
2. Involves next-sentence prediction (NSP). Assuming two consecutive sentences A and B, 50% of the time sentence B actually follows sentence A -- the other 50% of the time, sentence B is a random sentence.

- Fine-tuning involves passing task-specific inputs and labels to tweak model parameters end-to-end
- BERT can then be used as:
	- Single sentence and sentence-pair classification
	- Single sentence tagging
	- Q&A

# (7) Improving on BERT (2019+)
- Since BERT, several improvements have been made to make it lighter ([[DistilBERT]], ALBERT), optimize it further ([[RoBERTa]]).
- ...

# (8) Everything is text-to-text
- Towards the end of 2019, the [[T5|Text-to-Text Transfer Transformer]] (T5) ==introduced a unified framework that converts all text-based problems into a text-to-text format.==
- Thus, the input and output are text strings, making the single T5 model fit for multiple tasks!

![[Pasted image 20240326232857.png]]

The T5 uses the Transformer *encoder-decoder-based* structure which the authors found to work best for text-to-text. (Makes sense, given that it's basically *translation*, in a way)

With the unified format, *authors thoroughly explored the effectiveness of transfer learning in NLP*:
- Encoder-decoder architecture with a denoising objective worked best (vs a decoder-only transformer stack)
- Among the unsupervised objectives, masked language modeling (BERT-style) worked best (vs prefix language modeling, deshuffling, etc.)
- Training on a full dataset for 1 epoch worked better than training on a smaller dataset for multiple epochs.
- ==Fine-tuning all layers simultaneously performed best (vs unfreezing each layer and tuning it, adding adapter layers of various sizes)==
- Increasing model size had a bigger impact relative to increasing training time or batch size


![[Pasted image 20240326233218.png]]