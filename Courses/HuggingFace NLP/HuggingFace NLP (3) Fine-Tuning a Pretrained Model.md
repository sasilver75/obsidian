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
loss = model(**batch).loss  # We access the loss as an attribute on the 

```