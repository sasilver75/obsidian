https://huggingface.co/learn/nlp-course/chapter1/1
This course is about doing NLP using libraries from the HuggingFace ecosystem, including:
- Transformers
- Datasets
- Tokenizers
- Accelerate

Ch 1-4: Intro to main concepts of the Transformers library.
Ch 5-8: Basics of Datasets and Tokenizers, then Classic NLP tasks
Ch 9-12: Showing how Transformer models can be used to tackle tasks in Speech and Vision too!

****
Course Setup
![[Pasted image 20240619163423.png]]

NLP Tasks (non-exhaustive):
- Classifying sentence (sentiment analysis)
- Classifying each word in sentence (sentiment analysis)
- Generating text content (generation)
- Extracting an answer from text (question answering, information extracting)
- Generating a new sentence from input text (conditional generation)

Why's it challenging?
- Computers don't process information like humans do! Text needs to be processed in a way that enables the model to learn from it, and because language is complex, we need to think carefully about how this processing is done.

Working with Transformers in HuggingFace
```python
from transformers import pipeline

# Pipelines are a great and easy way to use models for inference; these pipelines are objects that abstract most of the complex code from the library, offering a simple API dedicated to several tasks, including NER/MLM/SA/QA/etc.
# There are currently ~20 or so accepted tasks; sentiment-analysis is one, returning a TextClassificationPipeline
classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
# [{'label': 'POSITIVE', 'score': 0.9598047137260437}]

# We can even pass several sentences!
pipeline(
	["I've been waiting for a HuggingFace course me whole life.", "I hate this so much."]
)
# [{'label': 'POSITIVE', 'score': 0.9598047137260437}, {'label': 'NEGATIVE', 'score': 0.9994558095932007}]

# By default, this pipeline selects a particular pretrained model finetuned for sentiment classification in english! The model is downloaded and then cached; if you rerun the command, the cached model will be used and there's no need to download the model again.
# Three main steps when you pass a text to a pipeline:
# 1) Text is preprocessed into a format that the model can understand
# 2) Preprocessed inputs are passed to the model
# 3) The predictions of the model are post-processed, so you can make sense of them.

# Let's try a zero-shot classification task where we need to classify texts that haven't been labeled.
# Teh zero-shot-classification pipeline allows us to specify which labels to use for the classification
classifier = pipeline("zero-shot-classification")
# We give it some text, and say: "These are three labels; classify it!"
clasifier("This is a course about the Transformers library", candidate_labels=["education", "politics", "business"])
# {'sequence': 'This is a course about the Transformers library', 'labels': ['education', 'business', 'politics'], 'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}

# This pipeline above is called "zero-shot" because we don't need to finetune the model on our data to use it -- it can directly return probability scores for ANY list of labels that you want!

# Let's try out some text generation, now
generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
# [{'generated_text': 'In this course, we will teach you how to understand and use '
                    'data flow and data interchange when handling user data. We '
                    'will be working with one or more of the most commonly used '
                    'data flows — data flows of various types, as seen by the '
                    'HTTP'}]
# Above; This seems like a ... fine generation, but I think it's unlikely that I'm going to use HF for text generation as opposed to some frontier or some specific finetuned open source language model.

# We can also use ANY MODEL FROM THE HUB in a pipeline!
# Note that we still have to pass a task string
generator = pipeline("text-generation", model="distillgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
[{'generated_text': 'In this course, we will teach you how to manipulate the world and '
                    'move your mental and physical capabilities to your advantage.'},
 {'generated_text': 'In this course, we will teach you how to become an expert and '
                    'practice realtime, and with a hands on experience on both real '
                    'time and real'}]

# Cool!
# We can refine oru search for a model by clicking on the language tasks, and pick a model that will generate text in another language, if we want.
# You can test models in your browser on their model page using 

```