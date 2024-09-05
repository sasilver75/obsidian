https://huggingface.co/learn/nlp-course/chapter7/1?fw=pt

----

In this chapter, we will tackle the following common NLP tasks:
- Token classification
- Masked language modeling (like BERT)
- Summarization
- Translation
- Causal language modeling pretraining (like GPT-2)
- Question answering

---------

## Token Classification
- The generic task of token classification encompasses any problem formulated as "attributing a label to each token in a sentence," such as:
	- [[Named Entity Recognition]] (NER)
	- [[Part-of-Speech Tagging]] (POS)
	- Chunking (Finding tokens that belong to the same entity; usually attributing one label (B-) to tokens at the beginning of a chunk, another (I-) to tokens inside the chunk, and a third label (0) to tokens that don't belong to any chunk)

In this section, we'll finetune a model (BERT) on a NER task.

Preparing the data
- We need a dataset suitable for token classification -- we'll use CoNLL-2003, which contains news stories from Reuters.

```python
from datasets import load_dataset

raw_dataset = load_dataset("conll2003")  # Download and cache the dataset
raw_datasets
DatasetDict({  # We can see that we have train/valid/test splits, with the columns for each dataset.
    train: Dataset({
        features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],  # See labels for NER, POS, Chunking tasks
        num_rows: 14041
    })
    validation: Dataset({
        features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
        num_rows: 3250
    })
    test: Dataset({
        features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
        num_rows: 3453
    })
})

# Let's look at one elment fomr the training set
raw_datasets["train"][0]["tokens"]
['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']

# and what do the NER tags look like for this?
raw_datasets["train"][0]["ner_tags"]
[3, 0, 7, 0, 0, 0, 7, 0, 0]  # The labels as integers
# We can access the corresponding labels by looking at the .features attr on our dataset
ner_feature = raw_datasets["train"].features["ner_tags"]
Sequence(feature=ClassLabel(num_classes=9, names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'], names_file=None, id=None), length=-1, id=None)  # B=Beginning, I=Inside, PER=Person, ORG=Organization, LOC=Location, MISC=Miscellaneous Entity

label_names = ner_feature.feature.names
label_names
['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

# Looking at some tokens and the ner_tags for them
'EU    rejects German call to boycott British lamb .'
'B-ORG O       B-MISC O    O  O       B-MISC  O    O'

# Let's preprocess the data
# As usual, our tokens need to be converted to token IDs before our model can make sense of them.
from transformers import AutoTokenizer
model_checkpoint = "bert-baes-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.is_fast  # True

# Let's tokenize a pre-tokenized input by jusing the is_split_into_words=True argument
inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
['[CLS]', 'EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'la', '##mb', '.', '[SEP]']

# As we can see, the tokenizer added the special tokens used by the model (CLS, SEP), and left most owrds untouched, but lamb was tokenized into two subwords, la and ##mb.
	# This introduces a mismatch between our inputs and the labels, since the list of labels only has 9 elements, whereas our intput now has 9+2+1=12 tokens! Accounting for the special tokens is easy (they're just at the beginning and end), but we need to make sure we align all the labels with proper words.
	# Becuase we're using afst tokenizer, we have access to the HF Tokenizer superpowers, meaning we can easily map each token to its correspodning word
inputs.word_ids()
[None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]

# With a little work, we can then expand our label list otm match the tokens!
	# We'll say that special tokens get a label of -100, which is an index that is ignored by the cross entropy loss function we'll use...
	# Then, each token gets the same label as the token that started the word it's inside, since they aren't part of the same entity. 
	# For tokens inside a word but not at the beginning, we replace the B- with I-
def align_labels_with_tokens(labels, word_ids):
	new_labels = []
	current_word = None
	for word_id in word_ids:
		if word_id != current_word:
			# start of new word
			current_word = word_id # The index of the word that this token came from, in the original sequence
			label = -100 if word_id is None else labels[word_id]  # The label (eg I-ORG) for this token
			new_labels.append(label)
		elif word_id is None:
			# Special token
			new_labels.append(-100)
		else:
			# Same word as previous token
			label = labels[word_id]
			# If label is B-XXX we change it to I-XXX
			if label % 2 == 1:
				label += 1
			new_labels.append(label)
	
	return new_labels

# Let's try it on our first sentence
labels = raw_datasets["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))
[3, 0, 7, 0, 0, 0, 7, 0, 0]  # Labels before (length=9 instead of 12)
[-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]  # Labels after (Added tokens for CLS and SEP, plus another for la+##mb)

# NOTE: Some researchers prefer to attribute only one label per word, and assign -100 to the other subtokens in a given word! This is to avoid long words that split into lots of subtokens contributing more heavily to the loss (since loss is computed on a per-token basis)
	# THIS SEEMS PRETTY REASONABLE TO ME!

# To preprocess our whole dataset, we tokenize all inputs and apply align_labels_with_tokens() on all labels.
# To take advantage of our fast tokenizer, it's best to tokenize lots of texts at the same time, so we'll write a function that processes a list of examples, and then use the Dataset.map method with batched=True to apply it.
def tokenize_and_align_labels(examples):
	tokenized_inputs = tokenizer(  # Tokenize the pre-tokenized exmamples.
		examples["tokens"], truncation=True, is_split_into_words=True
	)
	all_labels = examples["ner_tags"]
	new_labels = []

	for i, labels in enumerate(all_labels):
		word_ids = tokenized_inputs.word_ids(i)
		new_labels.append(align_labels_with_tokens(labels, word_ids))  # Expand the labels for the example as neede for CLS/SEP/split tokens

	tokenized_inputs["labels"] = new_labels  # Update the labels with the new labels
	return tokenized_inputs

# Note: WE haven't padded our inputs yet; we'll do that later when creating batches, using a data collator function.

tokenized_datsets = raw_datasets.map(
	tokenized_and_align_labels,
	batched=True,
	remove_columns=raw_datasets["train"].column_names
)

# That's the hardest part! Now that the data has been preprocessed, the actual training will look a lot like what we did in Chapter 3.

# Now let's finetune the mdoel with the Trainer API

# Collation
# We can't just use the DataCollatorWithPadding like in Ch3 because that only pads the inputs (inputIds, attention mask, token type Ids). Here, our labels need to be padded in the exact same way, so the inputs stay the same size. WE use -100 as a value, so the corresponding predictions are ingnoerd in the loss computation.
from transofrmers import DataCollatorForTokenClassification
collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Let's test it on a list of examples from our tokenized training set
batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
batch["labels"]
tensor([[-100,    3,    0,    7,    0,    0,    0,    7,    0,    0,    0, -100],
        [-100,    1,    2, -100, -100, -100, -100, -100, -100, -100, -100, -100]])

# Let's compare this to the labels for the first and second elements in in our datset
[-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]
[-100, 1, 2, -100]
# We can see that our send set of labels has been padded to the length of the first one, using -100s.

# Metrics
# To have the TRainer compute a metric every epoch, we'll need to define a compute_metric() function that takes the arrays of predictions and labels and returns a dictionary with the metric names and values
# The traditional framework to evaluate tkoen classification is seqeval
!pip install seqeval
import evaluate  # But then proceeds to use the evaluate library from HF?
metric = evaluate.load("seqeval") # Ah, then we load seqeval, interesting...
# This metric takes a list of labels as strings, rather than integers, so we need to fully-decode the predictions nad labels before passing them to the metric
labels = raw_datasetes["train"][0]["ner_tags"]
labels = [label_names[i] for i in labels]
labels
['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']

# And for demonstrations we can create fake predictions by just changing the value at index 2
predictions = labels.copy()
predictions[2]="0"
metric.compute(predictions=[predicitons], references=[labels])
{'MISC': {'precision': 1.0, 'recall': 0.5, 'f1': 0.67, 'number': 2},
 'ORG': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1},
 'overall_precision': 1.0,
 'overall_recall': 0.67,
 'overall_f1': 0.8,
 'overall_accuracy': 0.89}
 # WE get the precision/recall/f1 for each separate entity, as well as overall.
 # This compute_metrics fn takes the argmax of the logits to convert them to predictions, and then we have to convert both labels and predictions from integers to strings. WE remove values whre the label is -100, then pass the resutls to the metric.compute() method
import nupmy as np
def compute_metrisc(eval_predS):
	logits, labels = eval_preds
	predictions = np.argmax(logits, axis=-1)

	# remove ignored index (special tokens) and convert to labels
	true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
	true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
	all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

# The last thing we need to do is define a model to finetune!
from tokenizers import AutoModelForTokenClassification

# We need to pass some dictionaries containing mappings from id to label and vice versa, becuase our model doesn't know how many labels we have.
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k,v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
	model_checkpoint,
	id2label=id2label,
	label2id=label2id
)
mode.config.num_labels  # 9 ; that's what we're looking for!

# Let's finetune the model with a Trainer
from transformers import TrainingArguments

args = TrainingArguments(  # arguments relating to the training loop
	 "bert-finetuned-ner",
	 evaluation_strategy="epoch",
	 save_strategy="epoch",
	 learning_rate=2e-5,
	 num_train_epochs=3,
	 weight_decay=0.01,
	 push_to_hub=True
)
# You've seen most of this before; we set some hyperparameters (like LR, epochs, weight decay) and we specify push_to_hub=True to indicate we want to save the model and evaluate it at the end of every epoch.

from transformesr import Trainer
trainer = Trainer(
	model=model,
	args=args,
	train_dataset=tokenized_datasets["train"]
	eval_dataset=tokenized_datasets["validation"],
	data_collator=data_collator, # batching and padding
	compute_metrics=compute_metrics, # metrics each epoch
	tokenizer=tokenizer 
)
trainer.train()

# Once training is comlplete, we use push_to_hub to make sure we upload the most recent version of the model
trainer.push_to_hub(commit_message="Training complete") # this command returns the url, if you want to inspect it.
```



## Finetuning a Masked Language Model
- In some cases, you'll want to fine-tune LMs on your data before training a task-specific head.
- ![[Pasted image 20240624111106.png|300]]
- Domain adaptation technique from [[ULMFiT]]
`

```python
from transformers import AutoModelForMaskedLM

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# Let's see how many parames the model has
model.num_parameters()  # 67M  ; compare this to BERT, having 110M

# Masked Language Modeling
text = "This is a great [MASK]."
# For humans, we can imagine many possibilities for the [MASK] token, like "day", "ride", "painting"
# DistilBERT was pretrained on English Wikiepdia and BookCorpus, so we imagine its predictions for [MASK] to reflect those domains. 
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Nowe we can tokenize our text example, pass it to the model, extract the logits, and print out the top 5 candidates
import torch

inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits

# Let's find the location of [MASK] and extract its logit!
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
	print(f">>> {text.rpelace(tokenizer.mask_token, tokenizer.decode([token]))}")
'>>> This is a great deal.'
'>>> This is a great success.'
'>>> This is a great adventure.'
'>>> This is a great idea.'
'>>> This is a great feat.'

# We can see that this refers to everyday terms, which is perhaps not surprising given the foundation of English wikipedia.
# Let's change the domain to something more niche (highly polarized movie reviews) by finetuning (MLM fashion) on a movie review corpus

# We'll use the famous Large Movie Review Dataset (IMDb), which is a corpus of movie reviews often used to benchmark sentiment analysis models
from datasets import load_dataset
imdb_dataset = load_dataset("imdb")
imdb_dataset
DatasetDict({  # train, test, and unsupervised (unlabeled)
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
})

# Let's take a random sample by chaining the Datset.select and Dataset.shuffle functinos to create a random sample
sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))
for row in sample:
	print(f"\n'>>> Review: {row['text']}'")
    print(f"'>>> Label: {row['label']}'")

'>>> Review: This is your typical Priyadarshan movie--a bunch of loony characters out on some silly mission. His signature climax has the entire cast of the film coming together and fighting each other in some crazy moshpit over hidden money. Whether it is a winning lottery ticket in Malamaal Weekly, black money in Hera Pheri, "kodokoo" in Phir Hera Pheri, etc., etc., the director is becoming ridiculously predictable. Don\'t get me wrong; as clichéd and preposterous his movies may be, I usually end up enjoying the comedy. However, in most his previous movies there has actually been some good humor, (Hungama and Hera Pheri being noteworthy ones). Now, the hilarity of his films is fading as he is using the same formula over and over again.<br /><br />Songs are good. Tanushree Datta looks awesome. Rajpal Yadav is irritating, and Tusshar is not a whole lot better. Kunal Khemu is OK, and Sharman Joshi is the best.'
'>>> Label: 0'

'>>> Review: Okay, the story makes no sense, the characters lack any dimensionally, the best dialogue is ad-libs about the low quality of movie, the cinematography is dismal, and only editing saves a bit of the muddle, but Sam" Peckinpah directed the film. Somehow, his direction is not enough. For those who appreciate Peckinpah and his great work, this movie is a disappointment. Even a great cast cannot redeem the time the viewer wastes with this minimal effort.<br /><br />The proper response to the movie is the contempt that the director San Peckinpah, James Caan, Robert Duvall, Burt Young, Bo Hopkins, Arthur Hill, and even Gig Young bring to their work. Watch the great Peckinpah films. Skip this mess.'
'>>> Label: 0'

'>>> Review: I saw this movie at the theaters when I was about 6 or 7 years old. I loved it then, and have recently come to own a VHS version. <br /><br />My 4 and 6 year old children love this movie and have been asking again and again to watch it. <br /><br />I have enjoyed watching it again too. Though I have to admit it is not as good on a little TV.<br /><br />I do not have older children so I do not know what they would think of it. <br /><br />The songs are very cute. My daughter keeps singing them over and over.<br /><br />Hope this helps.'
'>>> Label: 1'

# Yep, these are certainly movie reviews! 0 is a negative review and 1 is a positive review.

# Let's preprocess our data!
# For both auto-regressive and MLM, a common preprocessing step is to concatenate all examples, and then split the corpus into chunks of equal size -- this is quite different from our usual approach, where we simply tokenized individaul examples.
	# Why do we do this? The idea is that individual examples might get truncated if they're too long, and that would result in losing information that would be useful for MLM.

# To get started, we'll tokenized our corpus as usual, but WITHOUT setting the truncation=True option in our tokenizer
def tokenize_function(examples):
	result = tokenizer(examples["text"])
	if tokenizer.is_fast
		# For a fast tokenizer, we'll use these later to do whole-word masking (after words have been split into subwords)
		result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
	return result

# Let's use batched=True to activate fast multithreading
tokenized_datasets = imdb_dataset.map(  # I suppose the remove_columns bit runs after the tokenize function
	tokenize_function, batched=True, remove_columns=["text", "label"]
)
tokenized_datasets
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['attention_mask', 'input_ids', 'word_ids'],
        num_rows: 50000
    })
})

# Since DistilBERT is a BERT-like model, we can see taht the encoded texts consist of the input_ids and attention_mask that we've seen in other chapters, as well as the word_ids we've added.
# Now that we've tokenized our movie reviews, the next step is to concat them all together and split the result into chunks... but how big should the hchunks be? Determined ultimately by the amount of GPU memory you have, but a good start is to see what your model's max context size is:
tokenizer.model_max_length  # 512 ; just like BERT

# So let's pick something a bit smaller that can fit in memory (so that it can run on cheap GPUs like Colab ones)
chunk_size = 128

# Let's look at the number of tokens per review from a few reviews in our tokenized datset
tokenized_samples = tokenized_datasets["train"][:3]
for idx, sample in enumerate(tokenized_samples["input_ids"]):
	print(f"'>>> Review {idx} length: {len(sample)}'")

'>>> Review 0 length: 200'
'>>> Review 1 length: 559'
'>>> Review 2 length: 192'

# We can then concat all of these examples together with a simple dictionary comprehension, as follows:

concatenated_examples = {
	 k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()  # Unsure about this use of sum
}
total_length = len(concatenated_examples["input_ids"])
print(f"Concatenated review length: {total_length}")  # 951

# The total length checks out, so let's split the concatenated reviews into chunks of the size given by chunk_size
chunks = {
	  k: [t[i:i+chunk_size] for i in range(0, total_length, chunk_size)]
	  for k,t in concatenated_examples.items()
}

for chunk in chunks["input_ids"]:
	print(f">>> Chunk length {len(chunk)}")
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 128'
'>>> Chunk length: 55'



```