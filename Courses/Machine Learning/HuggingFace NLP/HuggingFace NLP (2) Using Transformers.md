https://huggingface.co/learn/nlp-course/chapter2/1?fw=pt

The Transformers library was created to solve some of the problems associate with training, deploying, and using these models.
- ==Ease of Use==: Downloading/loading/using a SoTA NLP model is just two lines of code
- ==Flexibility==: All models are simple PyTorch `nn.Module` or TF `tf.keral.Model` classes
- ==Simplicity==: Hardly any abstraction are made across the library; the "All in one" file is a core concept; a model's forward pass is entirely defined in a single file, so that the code itself is understandable and hackable.
	- This is a unique feature that makes Transformers different from other ML libraries!

1. In this chapter we'll look an end-to-end example where we use a model and a tokenizer together to replicate the `pipeline` function we saw in the introduction.
2. Next, we'll discuss the model API, diving into the model and configuration classes, and how to configure it to process numerical inputs to output predictions.
3. Finally, we'll look at the tokenizer API, which is the other main component of the `pipelin` function, taking care of the first and last processing steps (handling the conversion of text to numerical inputs for the NN and back to text when needed).
4. Finally, how to handle sending multiple sentences through a model in a prepared batch, wrapping it all up with a closer look at the high-level `tokenizer` function.

![[Pasted image 20240619180606.png|300]]
This is what Pipelines do
![[Pasted image 20240619180620.png|300]]
This is what tokenizers do
- The `AutoTokenizer` class from Transformers has a nice .from_pretrained model which downloads and caches the configuration and vocabulary of a model.
![[Pasted image 20240619180736.png|300]]
Input Ids show the token ids of each token, and Attention Mask indicates where attention has been applied

![[Pasted image 20240619180809.png|300]]
The AutoModel API downloads and caches the configuration of a model, as well as the pretraind weights. It will only isntantiate the body of the model; the part left once the pretraining head is removed
- 2=batch size
- 16=sequence length
- 768=hidden dimensionality of the model

To get an output for our AutoModelForSequenceClassification class, which works the same, but it has a classification head
- There's one auto calss for each NLP task in the Transformer libarry.
![[Pasted image 20240619181015.png|300]]
These aren't probabilities yet; each model in the Transformer library uses Logits.
- The last step of the pipeline is post-processing, which in this case means converting logits to probabilities by applying a softmax activation function

Let's get into some code so that it makes a little more sense

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")  # This encompasses tokenization, model, preprocessing abilities
classifier(
	[
		"I've been waiting for a HUggingFace coursem y whole life!",
		"I hate this so much!"
	]
)
[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]

# Again, that pipeline groups together three steps: Preprocessing, model inference, and post-processing of reuslts

# Preprocessing with a Tokenizer
# Transformer's can't processs raw text directly, so first we need to convert the text inputs into numbers that the model can make sense of. To do this, we use a tokenzer, which:
	# Splits the input into words/subwords/symbols called tokens
	# Maps each token to an integer
	# Adds additional inputs (eg start/end tokens) that might be useful to the model

# All of this needs to be done in EXACTLY the same way as when the model was pretrained, so we need to download that information from the Model Hub; The AutoTokenizer class makes this easy, using the .from_pretrained method.
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Transformer models (eg in PyTorch) only accept TENSORS as inputs; to specify the type of tensors we want to get back, we can use the return_tensors argument (PyTorch, TensorFlow, or plain NumPy)
raw_inputs = [
	"I've been waiting for a HF course my whole life",
	"I hate this so much!"
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")  # PyTorch = "pt"
print(inputs)
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
# The output itself is a dictionary containing two keys, input_ids and attention_mask
	# input_ids contains two rows of integers (one for each sentence) that are unique identifiers 
	# attention_mask: Used when batching sequences together; indicates which tokens shoudl be attended to

# Now let's create our Model
# Transformers provides an AutoModel class which also has a .from_pretrained method
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

# This has the base Transformer encoder module; given some inputs, it outputs hidden states known as features... 
# These are usually used as inputs to another part of the model -- a head.
# The output has three dimensions: Batch Size, Sequence LEngth, Hidden Size (hidden dimension size)

outputs = model(**inputs)
print(outsputs.last_hidden_state.shape)
# torch.Size([2, 16, 768])

# The outputs of TRansformer models behave like namedtuples/dictionaries; you can access the elements by attributes or by key, or even by index if you know where the thing you're looking for is.

# Model heads are usually composed of one or a few linear layers
# AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForMultipleChoice, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoModelForTokenClassification
# In our examples, we need a model with a sequence classsification head, so we'll use...
from transformers import AutoModelForSequenceClassification
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.frmo_pretrained(checkpoint)
outputs = model(**inputs)
# But now the shape of our outputs is going to be different, with much lower dimensionality:
print(outputs.logits.shape)
# torch.Size([2,2])  # We havae output vectors containing two values (one per label), and since we have two sentences, it will be a 2x2

# Postprocessing the output
print(outputs.logits)
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)

# Recall that these are logits, not probabilities; to convert to probabilities, they need to go through a SoftMax layer.
import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1) # -1 means last dimension?...
print(predictions)
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
# These are recognizable probability scores that are positive and add to one!

# To get labels correspondign to each position, we can examine the model.config.id2label attribute
model.config.id2label
{0: 'NEGATIVE', 1: 'POSITIVE'}


# Let's look closer at using a Model, and the AutoModel class
# AutoModel and its relatives are simple wrappers over the weide variety of models available in the library -- they're clever wrappers that can automatically guess the appropriate architecture for your checkpoint, and instantiates a model with this architecture.
# If you know the TYPE of model you want to use, you can use the class that defines this architecture DIRECTLY, too.

from transformers import BerConfig, BertModel

# Building the config
config = BertConfig()  # Contains information on vocab size, hidden size, # hidden layers, positional embeddings, etc.

# Building the specific BERT model from the config
model = BertModel(config)

# When we initialize the model like we did above, it's still initialized with random values and will output gibberish -- it needs to be trained first. We COULD train it by hand, but that would require a lot of time and data. It's better to reuse models that have already been trained.
model = BertModel.from_pretrained("bert-base-cased")  # For this, we don't have to use BertConfig; it's initialized with all the weights of teh checkpoint model on the Model Hub that we indicate. The weights are downloaded and stored in your cache folder, which defaults to ~/.cache/huggingface/transformers (can be customized with HF_HOME environment variable)

# Saving a model is as easy as loading one:
model.save_pretrained("directory_on_my_computer")
# This saves both a config.json file and a pytorch_model.bin file
	# config.json file contains necessary attributes to build the mdoel architecture + metadata
	# pytorch_model.bin contains the state_dictionary of our model's weights

# Now that we have a model, let's make soem predictions! Ttransformer models can only process numbers, though
sequences = ["Hello!", "Cool.", "Nice!"]
# Our tokenizer converts each of these sequences into a list of numbers:
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
# This is a list of encoded sequences (list of lists), but we can convert it to a tensor:
import torch
model_inputs = torch.tensor(encoded_sequences)

# We can just pass this tensor of tokenized sequences to our model
output = model(model_inputs)

# Let's jump into Tokenizers now!
# Tokenizers can be word-based, subword-based, character-based, etc.
# It's important to be able to have a cutom token to represent words that are not in our vocabulary -- this is often known as the "unknown" token, often represented as "[UNK]" or "<unk>". It's generally a bad sign if you see a LOT of these tokens... but they're unavoidable, especially if you encounter words or characterse at test time that you didn't see during training time.
# The benefit of charactter-based tokens is that the vocabulary is smaller and there will be many fewer UNK tokens, but your sequence length will be longer, so autoregressive decoding will take longer, and each representation is less meaningful (can the model reconstruct the same meaning as in a word or subword-based tokenization strategy?)
# techniques for tokenization include BPE, WordPiece, SentencePiece, etc.

# Loading and saving tokenizers is as simple as it is with models (.from_pretrained(), and .save_pretrained())
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

tokenizer("Using a TRansformer network is simple")
{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

tokenizer.save_pretrained("directory_on_my_computer")

# The first step is to split the text into words (tokens); multiple rules can govenr this process
# The second step is to convert these oktens into numbers, so we can build a tensor out of them and feed them to the model. Tokenizers have VOCABULARIES, and we need to use the same vocabulary as when the mdoel was pretrained.

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("bert-baes-cased")
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)  # Why do we need .tokenize here, when for BertTokenizer we .__call__'d

print(tokens) # Ahh maybe because this gives the actual tokens, not the token ids
# ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
[7993, 170, 11303, 1200, 2443, 1110, 3014]
# These outputs, once converted to the tensor appropriate for the deep learning framework you're using, can be used as inputs to a model as seen earlier in this chapter.

# Decoding
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string) # Because we know the vocabulary of a tokenizer, it's easy to decode ids to strings!
'Using a Transformer network is simple'
# Above: Note that we convert the indices back to tokens, but also groups them together if they were part of the same word, making a readable sentence.

# Now let's talk about how we handle multiple sequences when we're doing inference!
# This is important when we're doing things like doing a forward pass on a batch of data dring training
# Questions: How do we handle sequences of different length? Are vocabulary indices the only inputs that allow a model to work well? Is there such a thing as too long of a sequence?

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a Hugging course my whole life."

tokens = tokenizer.tokenize(sequence)  # The actual tokens
ids = tokenizer.convert_tokens_to_ids(tokens)  # The token ids
input_ids = torch.tensor(ids)  # Turn them into a pytorch tensor

# This line will fail
model(input_ids)  # IndexError: Dimesnion our of range; expected ot bein range of [-1,,0], but got 1

# The problem si that we sent a SINGLE SEQUENCE to a model, whereas Transformer models expect multiple sequences by default.
tokenized_inputs = tokenizer(sequence, return_tensors="pt")  # This adds a dimension on top of it, which is great.
tokenized_inputs
tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102]])
# See the added dimension, above?

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor([ids]) # Us adding anothe dimension
output = model(input_ids)  # Produces logits

Input IDs: [[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012]]
Logits: [[-2.7276,  2.8789]]

# Batching is the act of sending multiple sentences through the model, all at once! If you only have one sentence, you can just build a batch with a single sequence:
batched_ids = [ids, ids]

# This is a batch of two identical sequences
# When we have a batch of multiple sentences, they're probably going to be of different lengths... but we need our tensor to be of a rectangular shape! So to work aronud this problem, we usually PAD the inputs!
batch_ids = [
	[200, 200, 200],
	[200, 200]
]

# Let's use padding to make these two sequences of indices into a rectangualr shape.
# Our tokenizer will use a specific padding token, which can be accessed by tokenizer.pad_token_id
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200,200,200]]
sequence_2_ids = [[200,200]]
batched_ids = [
	[200, 200, 200],
	[200, 200, tokenizer.pad_token_id],
]
print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)

tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
tensor([[ 1.5694, -1.3895],
        [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)
# Wait, something's wrong -- the logits in the second row should be the same as the logics for the second sentence, but we accidentally got different values!
# This is because Transformer models have layers that contextualize each token, taking into account the padding tokens.
# So we need to tell these attention layers to IGNORE the padding tokens!
# We do this using an ATTENTION MASK!

# Attention masks are tensors in the same shape as the input tensor IDs, filled with 0s and 1s.
batched_ids = [
	[200, 200, 200],
	[200, 200, tokenizer.pad_token_id],
]
attention_mask = [
	[1,1,1],
	[1,1,0]
]
outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
outputs
tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
# Now we get the "correct" outputs for the second sentence in the batch!
# Notice how the last value of the second sequence is a padding ID, which is a 0 value in the attention mask.

# With TRasnformres, there's a limit to the lengths of sequences that we can pass to models; most models handle sequences of up to 512 to 1024 tokens (longer, now) and will CRASH when asked to process longer sequeunces.
# You can either 1) Use a mdoel with a longer supported sequence length or 2) Truncate your sequences

sequence = sequence[:max_sequence_length]  # This is an easy way to truncate a sequence (but this sequence has already been tokenized! Doing this pre-tokenization will likely not be enough.)


# Let's put it all together, now!
from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)

# Here, model_inputs contains EVERYTHING necessary for a model to operate well -- for DistiliBERT, that includes the Input IDs as well as the Attention Mask. Other models that need more inputs will also have those output by their tokenizer object.

sequence = "I've been waiting for a HuggingFace course my whole life!"
model_inputs = tokenizer(sequence)
sequences = ["I've been waiting for a HuggingFace course my whole life!", "So have I!"]
model_inputs = tokenizer(sequences)

# We can also direct the tokenizer to pad for several objectives:
model_inputs = tokenizer(sequences, padding="longest")  # padded up to batch max sequence length
model_inputs = tokenizer(sequences, padding="max_length") # padded up to MODEL max sequence length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8) # padding up to 8

# We can even truncate sequences!
model_inputs = tokenizer(sequences, truncation=True)  # Truncates sequences longer than MODEL max seq length
model_inputs = tokenizer(sequences, truncation=True, max_length=8) # Truncates sequences longer than 8 tokens

# The tokenizer can also handle conversion to sspecific framework tensors, which we can then send directly to the model -- for example "pt" returns PyTorch tensors, and "np" returns NumPy arrays.
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Special Tokens
sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])
# If we examine this sequence of numbers, we notice that there was one added to the beginning and end of the sequence. 
tokenizer.decode(model_inputs["input_ids"])
"[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
# The tokenizer added the special CLS and SEP words as start and end tokens (BERT)

# Let's see one final time how we can handle multiple sequences (padding!), very long sequences (truncation!) and multiple types of tensors:
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretraiend(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)  # tokens is a dict (eg) with "input_ids" and others










```