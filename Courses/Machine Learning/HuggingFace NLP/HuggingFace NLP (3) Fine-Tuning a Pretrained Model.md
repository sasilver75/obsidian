In the previous chapter we learned how to use tokenizers and pretrained models to make predictions, but what if we wanted to fine-tune a pretrained model for our own dataset?
But what if we want to finetune a model for our specific use case? That's the subject of this!

## Processing the data

Here's how we might train a sequence classifier on one batch in PyTorch:

```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = [
	 "I've been waiting for a course like this!",
	 "This course is amazing!"
]

# For padding/truncation, we can pass a bool, str, or PaddingStrategy/TruncationStrategy
	# padding=True == "longest", padding the sequence to the longest sequence in the batch
	# trunctation=True == "longest_first", truncating to a maximum length specified by the max_length argument, or other to the maximum acceptible input length for the model.
batch: BatchEncoding = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# BatchEncoding holds hte output of a tokenizer's encoding method, and is derived from a Python dictionary.
batch["labels"] = torch.tensor([1,1])

optimizer = AdamW(model.parameters())  # This AdamW is from transformers, not from nn.optim
# The SequenceClassifierOutput class returns from our model.__call__ has the loss on it
	# See more here: https://huggingface.co/docs/transformers/en/main_classes/output
loss = model(**batch).loss  # We access the loss as an attribute on the ModelOutput
loss.backward() 
optimizer.step()
# Above: loss.backward and optimizer.step look identical to the vanilla PyTorch steps
```

This is cool, but training the model on two sentences isn't going to yield good results -- we need more data!
- WE'll use the MRPC (Microsoft Research Paraphrase Corpus, 2021), consisting of 5801 pairs of sentences, indicating whether they're paraphrases or not. It's a nice small dataset to experiment with.
	- It's one of th 10 datasets comprising the GLUE benchmark.

Let's download it!
```python
from datasets import load_dataset

# first positional parameter is "path", and second is "name"
raw_datasets = load_dataset("glue", "mrpc")  # download and cache dastsets; default location is ~/.cache/huggingface/datasets
raw_datasets
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})

# We can access each of tehse datasets via key, and then can access each pair of sentences in the dataset by indexing
raw_train_dataset = raw_datasets["train"]
raw_train_datset[0]
{'idx': 0,
 'label': 1,
 'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
 'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .'}

# Cool, we see that the labels are already integers, so no processing needed there... but what does  1 correspond to?
# We can inspect the features of our dataset to give us more information
raw_train_dataset.features
{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),  # So a 1 is "equivalent"
 'idx': Value(dtype='int32', id=None)}

# Let's preprocess the dataset by converting the text to numbers that the model can actually make sense of!
# We do this with a tokenizer; we can feed tokenizers either a sentence or a lsit of sentences:
from transformers import AutoTokenizer

checkpoint="bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datsets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datsets["train"]["sentence2"])

# But we're going to be passing the two sentences to the model as a pair... fortunately the tokenizer can also take a pair of sequences and prepare it the way our BERT model expects!
inputs = tokenizer("First sentence", "Second sentence")
inputs
{ 
  'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], # Interesting, this seems to indicate which part is the first sentence and which is the second!
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
# See the use of token_type_ids above, which we haven't talked about yet.
# We can decode the IDs in the input_ids key above back to words:
tokenizer.convert_ids_to_tokens(inputs(["input_ids"]))
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
# Above: See that this inserted a CLS token and two SEPs -- nice! The "1"s in token_type_ids start at "this", after the first [SEP]\
	# NOTE that if you have a different checkpoitn, you won't necessarily have this token_type_ids item in your tokenizer output dict -- for instance, DistilBERT doesn't reutnr them. They're only returned when teh model will know what to do with them, like BERT, who was pretrained using token type IDs (since BERT is trained with an additional objective of next sentence prediction)
	# In general, you don't really have to worry about all of the above; AS LONG AS YOU USE THE SAME CHECKPOINT FOR THE TOKENIZER AND THE MODEL, EVERYTHING WILL BE FINE, SINCE THE TOKENIZER KNOWS WHAT TO PROVIDE TO ITS MODEL!

# So let's feed the tokenizer a list of pairs of sentences by giving it the list of first sentences, then the list of secodn sentences.
tokenized_dataset = tokenizer(
	raw_datasts["train"]["sentence1"],
	raw_datasets["train"]["sentence2"],  # So it seems we can pass a singular string or a list of strings
	padding=True,
	truncation=True
)

# This works well, but has the disadvantage of returning a dictionary (with our keys of input_ids, attention_mask, and token_type_ids)
# This only works well if you have enough RAM to store your whole dataset during Tokenization!

```

Datasets from the HF Datasets library are stored as [[Apache Arrow]] files stored on disk, so you only keep the samples you ask for loaded in memory.
If we want to keep the data as a *dataset,* we can use the ==Dataset.map()== method, which works by applying a function on each element of the dataset... so let's define a function that tokenizes our inputs!
- Benefits: REsults of the funciton are cached, can apply both batching and multiprocessing to go faster, and doesn't load the whole dataset into memory, saving results one at a time(?)

```python
def tokenize_function(example):
	# Given a record dictionary, return the joint(?) tokenization of the two sentence features
	return tokenizer(example["sentence1"], example["sentence2"])
```

Note that this function also works if the example dictionary contains several samples, where each key is a list of sentences
- This allows us to use the ==batched=True== argument in our call to `Datasets.map()`, which will greatly speed up the tokenization!
- The tokenizer is actually backed by a tokenizer written in Rust from the Tokenizers library, which is very fast when we give it lots of inputs ast once.

Note that we're going to leave our the padding argument, because we haven't yet split our data into batches yet; it's more efficient to pad each batch individually, so that the longest sentence in your dataset doesn't dominate the padding requirements of every other sentence.
```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets

DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
})

# You can use multiprocessing when applying you processing function with map() by passing along a num_proc argument; we didn't do this here because Tokenizers library already uses multiple threads to tokenize our samples fster... but if you're not using a fast tokenizer backed by this library, this *could* speed up your processing.

# Now that we've tokenized all of our sentence pairs, we just neded to batch and do padding (a technique referred to as dynamic padding)
# The Tranformers library provides us with a ollation function DataCollatorWithPadding, which takes a tokenizer (to know which padding to use) and whether the model expects padding to be on the left or right, and will do everything we need.
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPaddding(tokenizer=tokenizer) # I assume it assumes default padding to be on the right side


# Let's now finetune our model with the Trainer API
# First, a recap:
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_fn(example):  # This fn can (when batched=True below) work on a dictionary where sentence1 and sentence2 are both lists
	return tokenizer(example["sentence1"], example["sentence2"])

tokenized_datsets = raw_datasets.map(tokenize_fn, batched=True)  # Tokenization of our sentences A,B into tokenized(A,B)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # A collation function to help us dynamically pad our batches

# Now let's get to our finetuning.

# First, we need to define our TrainingArguments class that will contain all of the hyperparameters that our Trainer will use for training and evaluation.
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")  # The only required argument is a directory where the trained model will be saved (incl. intermediate checkpoints)

# Now let's define our Model
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Unlike in Ch2, we'll get a warning after instantiating this model, because BERT wasn't pretrained on classifying pairs of sentences -- so the head of the pretrained model has been discarded, a new head has been added instead. We need to train this model.
from transformes import Trainer
trainer = Trainer(
	model
	training_args,
	train_dataset=tokenized_datasets["train"],
	eval_dataset=tokenized_datasets["validation"],
	tokenizer=tokenizer  
) # The default data_collator used by the Trainer will be a DataCollatorWithPadding, so we don't need to pass ours.

# Now we just need to call the .train() method to kick off fie-tuning!
trainer.train() 
# This will report the training loss every 500 steps (batches), but it won't tell you how well your model is performing, because:
	# We didn't tell the Trainer to evaluate during training by setting the evaluation_strategy="steps" (evaluating every eval_steps) or "epoch" (evaluate at the end of each epoch). 
	# We also didn't provide the Trainer with a compute_metrics() function to calculate a metric during said evaluation (otherwise the evaluation would have just printed the loss, which isn't a very intuitive number.)


# Let's build a useful compute_metrics() fn and use iet the next time we train
	# It takes an EvalPrediction object (named tuple with fields "predictions" and "label_ids"), and returns a dictionary mapping strings to floats, with the strings being the names of metrics returend, and the floats their values.
predictions = trainer.predict(tokenized_datasets["validation"])  # Note that we now use .predict on the trainer, which wraps our model.
print(predictions.predictions.shape, predictions.label_ids.shape)
(408, 2) (408,)  # predictions contains the logits for each element of the dataset passed to predict. We take the index with the maximum value on the second axis.

# The output of the predict() method is another named tuple with three fields: predictions, label_ids, and metrics
# Right now, this metrics key just contains the loss on the dataset (and some other misc metrics), but if we complete our compute_metrics() fn and pass it to the TRainer, that field will also contain the new metrics.

# Get our predictions from the output logits in predictions.predictions
preds = np.argmax(predictions.predictions, axis=-1)
# Now we can compare these preds to the labels, using some mtrics from the Evaluate library from HF
import evalaute
metric = evaluate.load("glue", "mrpc")  # We're loading the metrics associated with the MRPC dataset
metric.compute(predictions=preds, references=predictions.label_ids)
{'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}

def compute_metric(eval_preds):
	metric = evaluate.load("glue", "mrpc")
	logits, labels = eval_preds  # Cool that we can destructure like that
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=preditions, references=labels)

# Now that we have this, we can recreate our trainer
training_args = TrainingArgumemnts("test-trainer", evaluation_strategy="epoch")  # Report metrics every epoch
model = AutoModelForSequenceClassification.from_pretraiend(checkpoint, num_labels=2) # See num_labels

trainer = Trainer(
	model,
	training_args,
	train_dataset=tokenized_datasets["train"],
	eval_dataset=tokenized_datasets["validation"],
	data_collator=data_collator,  # I'm not sure that this is required; I think our Trainer by default will use a DataCollatorWithPadding
	tokenizer=tokenizer,
	compute_metrics=compute_metrics
)

# And now we can train our model again!
trainer.train()
```

## A Full Training (In PyTorch)
Let's now look to do a full training in PyTorch, *without* using the Trainer API!

Here's a summary of our progress so far:
```python
from datasets import load_dataset
from transforers import AutoTokenizer, DataCollatorWithPAdding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
	return tokenizer(example["sentence1"], example["sentence2"])

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DatacollatorWithPadding(tokenizer=tokenizer)  # So that it knows to pad on left or right, etc, and *which* padding tokens to use.
```

Now let's prepare for training in PyTorch! We'll need
- Postprocessing to our tokenized_datasets to take care of things that the Trainer did for us automatically
	- Remove columns corresponding to unexpected values (sentence1, sentence2)
	- Rename the column label to labels
	- Set the format of datasets so they return PyTorch tensors instead of lists
- Dataloaders we can use to iterate over our dataset in batches

Our `tokenized_datasets`  has a method for each of these steps!
```python
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
["attention_mask", "input_ids", "labels", "token_type_ids"]

# Now we can define our PyTorch DataLoaders

train_dataloader = DataLoader(
	  tokenized_dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator  # DataCollatorWithPadding for dynamic padding during batch construction
)
eval_dataloader = DataLoader(  # Don't need to shuffle in Eval
	 tokenized_dataset["test"], batch_size=8, collate_fn=data_collator
)

# We can manually inspect a batch like this:
batch = next(iter(train_dataloader))
{k:v.shape for k,v in batch.items()}
{'attention_mask': torch.Size([8, 65]),
 'input_ids': torch.Size([8, 65]),
 'labels': torch.Size([8]),
 'token_type_ids': torch.Size([8, 65])}
# Nice, we can see that the padding worked because the dimensionalities all match on the 65


# now that we're done with preprocessing, we can instntiate our model
from transformer simport AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

outputs = model(**batch)  # All 🤗 Transformers models will return the loss in output.loss when `labels` are provided, and we also get the logits in output.logits (two for each input in our batch, so a tensor of size 8 x 2).

# Cool! But that model isn't trained yet (recall numclasses=2 issue)
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5) # This is just going to be our starting LR

# The learning rate scheduler used by default is just a linear decay from the maximum value to 0; to preoprl define it, we need to know the number of training steps we take with is n_epochs*n_batches_per_epoch; Trainer uses 3 epochs by default, so we'll follow that
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
	 "linear",
	 optimizer=optimizer,  # This optimizer instance already has a starting LR set on it
	 num_warmup_steps=0,
	 num_training_steps=num_training_steps
)

print(num_training_steps)  # 1377

# Training Loop
import torch
device = torch.device("cuda") if torch.cuda.is_avaialble() else torch.device("cpu") # I think you can just use the strings of "cuda" or "cpu" here too...
# Move the model to GPU memory; we'll have to move the data there too
model.to(device)  

from tqdm.auto import tqdm  # The arabic "_taqaddum_" means "Progress"; it's a progress bar library

progresS_bar = tqdm(range(traininG_steps))

model.train()  # Put the model in training model
for epoch in range(num_epochs):
	for batch in train_dataloader:
		batch = {k: v.to(device) for k,v in batch.items()}  # Put the batch on the GPU
		outputs = model(**batch)  # Do inference
		loss = outputs.loss # Recall that our results always just have loss on them! (I don't think WE had to define the loss fn)
		loss.backward()  # Notice that they use the style of zeroing the gradients _after_

		optimizer.step()
		lr_scheduler.step()  # Step your LR scheduler too
		optimizer.zero_grad()  # Notice that their preference here is to zero grad after the update (in preparation of next epoch). I think I prefer the inverse.
		progress_bar.update(1)  # Update our tqdm progress bar too

# And now the evaluation loop,. using a metric from the Evaluate library!
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()  # Put it in train mode; I notice that they don't use torch.inference_mode() here... They instead use torch.no_grad, which I think is dispreferred
for batch in eval_dataloader:
	batch = {k:v.to(device) for k in batch.items()}
	with torch.no_grad():  # Instead, we shoudl be using torch.inference_mode()!
		outputs = model(batch)

	logits = outputs.logits  # logits
	predictions = torch.argmax(logits, dim=-1)  # Pull the class predictions from the logits
	metric.add_batch(predictions=predictions, references=batch["labels"])  # Passing yhat and y

metric.compute()  # It seems we manually call compute (when not using Trainer) on our metric, after (remembering to) add a number of batches to it.

# The loop we defined above works fine on a single CPU or GPU, but by using the Accelerate library, we can enable DISTRIBUTED TRAINING on multiple GPUs or TPUs! 
# Here's what that looks like:
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_epochs = 3
num_training_steps = num_epochs * len(train_datloader)

progress_bar = tqdm(range(num_training_steps))


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2).to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)

lr_scheduler = get_scheduler(
  "linear",
  optimizer=optimizer,
  num_warmup_steps=0,
  num_training_steps=num_training_steps
)

model.train()
for epoch in range(num_epochs):
	for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # Map our data to device 
        outputs = model(**batch)  # forward pass
        loss = outputs.loss # extract loss
        loss.backward()  # get gradients
        optimizer.step()  # update params
        optimizer.zero_grad()  # zero out grads
        lr_scheduler.step()  # remember to step lr_sched
        progress_bar.update(1)  # remember to step progress bar
```

If we wanted to use Accelerate, the only changes we need to make are:
![[Pasted image 20240620192649.png]]
That's hardly anything!
- We create an Accelerator object that will look at the environment and initialize the proper distributed setup.
	- Accelerate handles device placement for you, so you can remove the lines that put the model on the device.
- The main bulk of the work is done in the line that sends the dataloaders, model, and optimizer to `accelerator.prepare()`, which wraps those objects in a container and returns them to make sure your distributed training works as intended.
- The remaining change is just removing the line that puts the batch on the device (not needed), and replacing the loss.backard() with accelerator.backward(loss)

You put your training code in a `train.py` file, and then run `accelerate config`, which prompts you to answer some questions, dumping your configuration into a file, and then you can run `accelerate launch train.py`, which will launch the distributed training.
- In a notebook, you just paste the code into a `training_function` fn, and then run a last cell with `accelerate.notebook_launcher(training_function`

