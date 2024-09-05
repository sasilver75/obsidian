https://huggingface.co/learn/nlp-course/chapter6/1?fw=pt

------

When we're using a pre-trained model to complete a task, we've so far learned that we need to use the same tokenizer that the model was pretrained with -- but what if we want to train a model from scratch?
- Using a tokenizer that was pretrained on a corpus from another domain or language is typically suboptimal.
- So let's learn how to build a brand new tokenizer on a corpus of texts, so that it can then be used to pretrain a language model!

Topics we'll cover include:
- How to train a new tokenizer ***similar*** to the one used by a given checkpoint on a new corpus of texts
- The special features of fast tokenizers in HF `Tokenizers`
- The differences between the three main subword tokenization algorithms used in NLP today
- How to build a tokenizer from scratch using `Tokenizers`, and training it on some data.

----

We saw that most Transformer models use some sort of *subword tokenization algorithm*. 

To identify which subwords are of interest and occur mostly frequently in the corpus at hand, a tokenizer needs to be trained on a corpus (the exact rules that govern this training depend on the tokenization algorithm used).

🤗 Tokenizers  lets us train a new tokenizer with the same *characteristics* as an existing one using `AutoTokenizer.train_new_from_iterator()`; to see this  in action, let's say we want to train GPT-2 from scratch but in another language besides English -- like Python!

```python
from datasets import load_dataset

raw_datasets = load_dataset("code_search_net", "python")
raw_datasets["train"]
Dataset({
    features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 
      'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 
      'func_code_url'
    ],
    num_rows: 412178
})

# We can see that the dataset separates dosctrings from code, and suggests a tokenization of both. We'll just use the "whole_func_string" column to train our tokenizer; it looks like:
print(raw_datasets["train"][123456]["whole_func_string"])
# def handle_simple_responses(
#       self, timeout_ms=None, info_cb=DEFAULT_MESSAGE_CALLBACK):
#     """Accepts normal responses from the device.
# 
#     Args:
#       timeout_ms: Timeout in milliseconds to wait for each # response.
#       info_cb: Optional callback for text sent from the bootloader.

# Let's transform the dataset into an _iterator_ of lists of texts. Using lists of texts will help our tokenizer go faster (training on batches of texts instead of processing individual texts one by one).
# Because it's lazy, the following generator doesn't fetch any elements frmo the dataset, just creates a generator object (which is an iterable) that we can use in a Python for loop
training_corpus = (  # Generator yielding batches from a dataset
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)  # Recall that slicing a Dataset (eg to a slice of length 2) results in a dictionary where each key's value is a list of length 2.

# The problem with a generator expression is that it returns a single generator object, which can only be iterated through once before becoming enhausted.
# This is why we define a function that returns a generator instead!
def get_training_corpus():
	return (
		raw_datasets["train"][i:i+1000]["whole_func_string"]
		for i in range(0, len(raw_datasets["train"]), 1000)
	)

# Alternatively, we could define our generator inside a for loop by using the yield statement
def get_training_corpus():
	dataset = raw_datsets["train"]
	for start_idx in range(0, len(dataset), 1000):
		samples = dataset[start_idx: start_idx+1000]
		yield samples["whole_func_string"]
```

Now that we have a way to lazily retrieve batches of data from our dataset to train a tokenizer, let's talk about training a tokenizer like the one used in GPT-2.
- It's probably smart to not start entirely from scratch, so that we don't have to specify many details. HF Tokenizers makes it easy to just change the vocabulary and training corpus, using an already-trained tokenizer's other settings.

```python
from transformers import AutoTokenizer

# This is an example of using the ACTUAL gpt-2 "old" tokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

example = '''def add_numbers(a,b):
	"""Add the two numbers 'a' and 'b'."""
	return a + b
'''

tokens = old_tokenizer.tokenize(example)
tokens
['def', 'Ġadd', '_', 'n', 'umbers', '(', 'a', ',', 'Ġb', '):', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo',
 'Ġnumbers', 'Ġ`', 'a', '`', 'Ġand', 'Ġ`', 'b', '`', '."', '""', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']

# We can see that the tokenizer has a few special symbols (like Ġ and Ċ, which denote spaces and newlines respectively). 
# We can see also that it's not super efficient; 
	# the tokenizer returns individual tokens for each space, when it could group together indentation levels (rather than having 4-8 spaces). 
	# It also split the function name a bit weirdly, not being used to seeing words with the _ character.

# Let's train a new tokenizer on our dataset to see if it solves these issues!
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)  # 52000 is the vocab size that we're setting

# Note that AutoTokenizer.train_new_from_iterator only works if the tokenizer that you're using is a "fast" tokenizer (Rust).

# Let's see how it tokenizes the same example?
tokens = tokenizer.tokenize(example)
tokens
['def', 'Ġadd', '_', 'numbers', '(', 'a', ',', 'Ġb', '):', 'ĊĠĠĠ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`',
 'a', '`', 'Ġand', 'Ġ`', 'b', '`."""', 'ĊĠĠĠ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
 # Here again we see the special G and C symbols denoting spaces and newlines, but we also see that our tokenizer has learned some  tokens that are highly specific to a corpus of Python functions! Like CGGG represents an indentation, and G""" represents the three quotes that start a docstring. It also corectly-split our function name on _. Quite a compact representation, comparatively!

# To make sure we can use it later, we need to save our tokenizer!
tokenizer.save_pretrained("code-search-net-tokenizer")

# This creates a new folder named  code-search-net-tokenizer  containing all of the files the tokenizer needs to be reloaded. You can also upload to your account!
tokenizer.push_to_hub("code-search-net-tokenizer")

# This will creaet a new repository in your namespace with the name code-search-net-tokenizer, containing the tokenizer file. You can then load it from anywhere with the from_pretrained() method!
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
```

## Fast Tokenizers' Special Powers
- The output of a tokenizer isn't a simple Python dictionary -- we get a special `BatchEncoding` object, which subclasses dictionary.
- (Skipping this section, it's boring and mostly marketing about under-the-hood features that I don't give a fuck about.)

## Fast Tokenizers in the QuestionAnswering pipeline
- The QA task deals with very long contexts that often end up being truncated.

We saw in Ch1 that we can use a QA pipeline to get the answer to a question
```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""

question = "Which deep learning libraries back 🤗 Transformers?"
question_answerer(question=question, context=context)
{'score': 0.97773,
 'start': 78,
 'end': 105,
 'answer': 'Jax, PyTorch and TensorFlow'}

# Unlike the other pipelines that can't truncate and split texts that are longer than the maximum length accepted by the model, THIS pipeline can deal with very long contexts and will return the answer to the question, even if it's at the end!
long_context = """
🤗 Transformers: State of the Art NLP

🤗 Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction,
question answering, summarization, translation, text generation and more in over 100 languages.
Its aim is to make cutting-edge NLP easier to use for everyone.

...

Why should I use transformers?

1. Easy-to-use state-of-the-art models:
  - High performance on NLU and NLG tasks.
  - Low barrier to entry for educators and practitioners.
  - Few user-facing abstractions with just three classes to learn.
  - A unified API for using all our pretrained models.
  - Lower compute costs, smaller carbon footprint:

...

4. Easily customize a model or an example to your needs:
  - We provide examples for each architecture to reproduce the results published by its original authors.
  - Model internals are exposed as consistently as possible.
  - Model files can be used independently of the library for quick experiments.

🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""

question_answerer(question=question, context=long_context)
{'score': 0.97149,
 'start': 1892,
 'end': 1919,
 'answer': 'Jax, PyTorch and TensorFlow'}
```


So how does it do all this?
- Like with any other pipeline, we start by tokenizing our input, and then send it through the model.
- The default model for the question-answering pipeline is `distilbert-base-cased-distilled-squad`:

```python
from transformers import AutoTokenizer, AutoModelForQuesiotnAnswering

checkpoint = "distilbert-base-cased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Note that we tokenize the question and the context as a pair, with the question first.
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# Models for question answering work a little differently from the models we've seen up to now. 
# The model has been trained to predict the index of the token starting the answer (here 21) and the index of the token where the answer ends (here 24). This is why the models don't return one tensor of logits, but two (one for the logits corresponding to the start token of the answer, one for the logits corresponding to the end token of the answer).

start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)
torch.Size([1, 66]) torch.Size([1, 66])

# To convert these logits into probabilities, we apply a softmax function. But before that, we need to make sure the indices aren't part of the context! 
# Our input is [CLS] question [SEP] context [SEP], so we need to mask the tokens of the question as well as the [SEP] token. We keep the CLS token, however, as some models use it to indicate that the answer isn't in the context.

# Since we're going to be softmaxing afterwards, we can replace the logits we want to mask with a large negative number, eg -10000
import torch

sequence_ids = inputs.sequence_ids()  # inputs=tokenization output

# Mask everything except the token from the context
mask = [i != 1 for i in sequence_ids]
# Unmask the CLS otken
mask[0] = False
mask = torch.tensor(mask)[None]  # ? What does this do? It seems like it basically unsqueezes a dimension, turning our [N] shape into [1, N] shape. 
  
start_logits[mask] = -10000  # Indexing using a boolean tensor is just like a filter; it returns a tensor of length == num(Trues) in tensor
end_logits[mask] = -10000

# Now that we've properly masked the logits corresponding to positions that we DON'T want to predict, we can apply the softmax:
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]
```

I've lost interest in this example.

## Normalization and pre-tokenization

Let's look at the preprocessing that each tokenizer applies to text 
![[Pasted image 20240623223504.png|300]]
The normalization step involves some general cleanup, such as removing needless whitespace, lowercasing, and/or removing accents. If you’re familiar with [Unicode normalization](http://www.unicode.org/reports/tr15/) (such as NFC or NFKC), this is also something the tokenizer may apply.

Transformers' `tokenizer` class has an attribute called backend_tokenizer that gives access to the underlying tokenizer from the Tokenizers library:
```python
from transformesr import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))
class <'tokenizers.Tokenizer'>

# The normalizer attribute of the tokenizer object has a normalize_str() method that we can use to see how the normalization is performed.
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
'hello how are u?'
# In this eample with our bert-base-uncased checkpoint, the noermalization applied lowercasing and removed the accents.

# The next step is pre-tokenization; we can again access the .pre_tokenize_str method of the pre_tokenizer attribute of the tokenizer object
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
[('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]
# Notice that the tokenizer keeps track of the offsets of tokens. Note it also ignored the fact that there were two spaces.
# Some other tokenizers (T5) might split on whitespace, but NOT punctuation, for example.
```

(Notes on Unigram, BPE, WordPiece added to respective notes)

## Building a Tokenizer, Block by Block
- More precisely, the library is built around a central `Tokenizer` class with the building blocks regrouped in submodules.

- `normalizers`: Contains all possible types of Normalizers you can use
- `pre_tokenizers`: Contains all possible types of PreTokenizers you can use.
- `models`: Contains the various types of Model you can use, like BPE, WordPiece, Unigram
- `trainers`: Contains all the different types of Trainer you can use to train your model on a corpus (one per type of model)
- `post_processors`: Contains the various types of PostProcessors you can use.
- `decoders`: Contains the various types of Decoder you can use to decode the outputs of tokenization.

To train our new tokenizer, we will use a small corpus of text. The steps for acquiring the corpus are similar to the ones we took at the beginning of the chapter, but this time using the WikiText-2 dataset.

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

# Generator that will yield batches of 1,000 texts, which we will use to train the tokenizer.
def get_training_corpus():
	for i in range(0, len(dataset), 1000):
		yield dataset[i: i+1000]["text"]

# Generating a text file containing all texts/inputs from WikiText-2 that we can use locally:
with open("wikitext-2.txt", "w", encoding="utf-8"):
	for i in range(len(dataset)):
		f.write(dataset[i]["text"] + "\n")

# Let's now build a WordPiece tokenizer frmo scratch by creating a Tokenizer object with a model, then set its normalizer, pre_tokenizer, post_processor, and decoder attributes to the values we want.
from tokenizers import (
	decoders,
	models,
	normalizers,
	pre_tokenizers,
	processors,
	trainers,
	Tokenizer
)
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# We have to specify the unk_token so that the model knows what to return when it encounters characters that it hasn't seen before. We can also specify the vocab for our model (but we're going to train our model, so we don't know that), max_input_chars_per_word (words longer than the value will be split).

# (1) The first step of tokenization is normalization, so let's begin with that. Since BERT is so widely used, theres a BertNormalizer with teh classic options we can set for BERT: lowercase and strip_accents, which are self-explanatory, and clean_text to remove all control characters and replace repeating spaces with a single one, and handle_chinese_chars which place spaces around chinese characters.
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

# Generally speaking, when building a new tokenizer, you won't have access to such a handy normalizer already implemented in the Tokenizers library  -- so let's see how to create the BERT normalizer by hand:
tokenizer.normalizer = normalizers.Sequence(
	[normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
) # The order matters in this sequence

# We're also using an NFD Unicode normalizer, as otherwise the StripAccents normalizer won't properly recognize the accented characters and thus won't strip them out.
print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
hello how are u?

# (2) Next, let's talk about the pre-tokenization step. There's a prebuilt BertPreTokenizer that we can use:
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

# or we can built it from scratch
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# Note that the Whitespace pre-tokenizer splits on both whitespace and all characters that aren't alphanumeric or underscores.
# If you only wanted to split on whitespace, you should use the WhitespaceSplit pre-tokenizer instead.

tokenizer.pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
[('Let', (0, 3)), ("'", (3, 4)), ('s', (4, 5)), ('test', (6, 10)), ('my', (11, 13)), ('pre', (14, 17)),
 ('-', (17, 18)), ('tokenizer', (18, 27)), ('.', (27, 28))]

# Like with normalizers, you can use a Sequence to compose several pre-tokenizers.
pre_tokenizer = pre_tokenizers.Sequence(
	[pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)
pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
[('Let', (0, 3)), ("'", (3, 4)), ('s', (4, 5)), ('test', (6, 10)), ('my', (11, 13)), ('pre', (14, 17)),
 ('-', (17, 18)), ('tokenizer', (18, 27)), ('.', (27, 28))]

# (3) Next, we can run the inputs through the tokenization model! We've already specified our model in the initialization, but we still need to train it, which requires a WordPieceTRainer.
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)  # It doesn't seem like we've passed any part of our tokenizer that we've been constructing yet

# As well as specifying the vocab_size and special_tokens, we can set the min_frequency (the number of times a token must appear to be included in the vocabulary) or change the continuing_subword_prefix (if we want to do something different from ##).

# To train our model using the iterator we defined earlier, we just execute this command:
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
# This is interesting use of trainer; for the modle finetuning section, iirc the trainer wrapped the model; here, the tokenizer method is given the trainer.

# We can also just train the model using a text file
tokeinzer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train(["wikitext-2.txt"], trainer=trainer)

# After tarining, we can test the tokenizer on a text by calling the .encode() method:
encoding = tokenizer.encode("Let's test this tokenizer.")
encoding.tokens
['let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.']''

# The encoding obtained is an `Encoding`, which contains all of the necessary outputs of the tokenizer in its various attributes (ids, type_id, tokens, offsets, attention_mask, special_tokens_mask, and overflowing).

# (4) The last step in the tokenization pipeline is post-processing -- we need to add the CLS token at the beginning and the SEP otken at the end.
# For this, we use a TemplateProcessor -- but we first need to know the IDs of the CLS and SEP tokens in the vocabulary.
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)  
(2,3)

# To write the template for TemplateProcessor, we specify how to treat a single sentence and a pair of sentences. For both, we write the special tokens we want to use; the first sentence is represented by $A, while the second sentence is represented by $B. 
# For each, we specify the corresponding token type ID after a colon. # This is the classic BERT template
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",  # :TokenTypeID
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",  # $A and $B
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)], # IDs of special tokens, so they can be properly converted
)

# Testing for a single sentence
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.', '[SEP]']

# Testing for a pair of sentences
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
print(encoding.tokens)
print(encoding.type_ids)
['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '...', '[SEP]', 'on', 'a', 'pair', 'of', 'sentences', '.', '[SEP]']
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# (5) The last step is to include a decoder
tokenizer.decoder = decoder.WordPiece(prefix="##")

# Let's test it on our previous encoding:
tokenizer.decode(encoding.ids)
"let's test this tokenizer... on a pair of sentences."


# Great! Now we can save our tokenizer in a single JSON file like this:
tokenizer.save("tokenizer.json")
# And later reload that file on a Tokenizer object with the .from_file() method:
new_tokenizer = Tokenizer.from_file("tokenizer.json")

# To use this tokenizer in Transforms, we have to wrpa it in a `PreTrainedTokenizerFast`
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
	tokenizer_object=tokenizer,
	unk_token="[UNK]",
	pad_token="[PAD]",
	cls_token="[CLS]",
	sep_token="[SEP]",
	mask_token="[MASK]",
) 

# When we use a specific tokenizer class (like BertTokenizerFast), you only need to specify the special tokens that are different frmo the defalut ones (here, none):
from transformers import BertTokenizer

wrapped_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
```

Then they have an example of building a BPE tokenizer from scratch too -- I'm going to skip that.