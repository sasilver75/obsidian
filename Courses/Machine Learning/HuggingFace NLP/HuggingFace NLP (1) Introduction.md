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
# You can test models in your browser on their model page, which uses the HuggingFace Inference API behind the scenes.

# Let's try a fill-mask pipeline, where the idea of the task is to fill in th eblanks in a given text:
unmasker = pipeline("fill-mask")
# It's important to understand what the mask token looks like for your model
unmasker("This course will teach you all about <mask> models.", top_k=2)
[{'sequence': 'This course will teach you all about mathematical models.',
  'score': 0.19619831442832947,
  'token': 30412,
  'token_str': ' mathematical'},
 {'sequence': 'This course will teach you all about computational models.',
  'score': 0.04052725434303284,
  'token': 38163,
  'token_str': ' computational'}]

# Pretty good! The top_k argument in this pipline will control how many possibilities you want to have displayed. Note that the model fills the special <mask> word, which is often referred to as a mask token. Check the mask word for the model in the Model page.

# Task: Named Entity Recogniiton
ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at HuggingFace in Brooklyn")
[{'entity_group': 'PER', 'score': 0.99816, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
 {'entity_group': 'ORG', 'score': 0.97960, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
 {'entity_group': 'LOC', 'score': 0.99321, 'word': 'Brooklyn', 'start': 49, 'end': 57}
]
# Cool! We can see that it identified a person, organization, and location! Nice. But not necessarily the relationship between them ;) The grouped_entities=True tells the pipeline to regroup together parts of the sentence that correspond to the same entity (eg Hugging + Face as a single organization... or even Sylvain being tokenized as S ##yl ##va and ##in, bu then in the post-processing step regroping these pieces.)

# Next ask: Question Answering
question_answerer = pipeline("question-answering")
question_answerer(
	question="Where do I work?", # Interesting that we use "I" here
	context="My name is Sylvain and I work at HuggingFace in Brooklyn"
)
{'score': 0.6385916471481323, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}

# Task: Summarization
summarizer = pipeline("summarization")
summarizer("""
	America has changed dramatically during recent yers. Not only ... {more text continues}
""")
[{'summary_text': ' America has changed dramatically during recent years . The '
                  'number of engineering graduates in the U.S. has declined in '
                  'traditional engineering disciplines such as mechanical, civil '
                  ', electrical, chemical, and aeronautical engineering . Rapidly '
                  'developing economies such as China and India, as well as other '
                  'industrial countries in Europe and Asia, continue to encourage '
                  'and advance engineering .'}]
# See that this captured some of the interesting parts of the sentence.


# Task: Translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")  # A fr->en MT model using Claude Opus?
translator("Ce cours est produit par Hugging Face.")
[{'translation_text': 'This course is produced by Hugging Face.'}]
# Nice translation, thanks!
```

Not that we've seen some of the basics of pipelines in `transformers`, let's look a little closer at Transformers generally!
- Introduced in June 2017
- GPT in June 2018
- October 2018: BERT
- Feb 2019: GPT-2
- Oct 2019: DistilBERT
- Oct 2019: BART and T5
- May 2020: GPT-3
- ...

These fall into mostly three categories:
- GPT-like: [[Decoder-Only Architecture]]
- BERT-like: [[Encoder-Only Architecture]]
- BART/T5-like: [[Encoder-Decoder Architecture]]

Transformers are language models!
- Trained on large amount of raw text in a self-supervised fashion, where the objective is automatically computed from the inputs of the model.
- Causal Language Modeling vs Masked Language Modeling

Training large models is expensive in terms of dollars and carbon impact, so it's important that we be able to share language models; sharing trained weights and building on top of already trained weights is an important facet of open source collaboration!

(Skipping Transformer Architecture information)

Even the biggest pretrained models come with limitations; researchers sometimes scrape all the content they can find, which includes the worst of what's available on the internet. Our models are biased (not inherently, but in actuality)

```python
unmasker = pipeline("fill-mask", model="bert-base-uncased")
result_m = unmasker("This man works as a [MASK].")
result_w = unmasker("This woman works as a [MASK].")
['lawyer', 'carpenter', 'doctor', 'waiter', 'mechanic']
['nurse', 'waitress', 'teacher', 'maid', 'prostitute']
```

