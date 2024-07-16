Link: https://huggingface.co/blog/vlms

---------

[[VLM|Vision-Language Model]]s are models that learn simultaneously from images and texts to tackle many tasks, from visual question answering to image captioning.

They are a type of generative model taking image and text inputs, generating text outputs.

Uses cases include:
- Chatting about images
- Image recognition via instructions
- Visual question answering
- Document understanding
- Image captioning
- ...

Some vision-language models can also capture spatial properties in an image, outputting bounding boxes or segmentation masks when prompted to detect or segment a particular subject, or they can localize different entities or answer questions about their relative or absolute positions.

There's a lot of diversity within the existing set of large vision language models, the data they were trained on, how to encode images, and their capabilities.

![[Pasted image 20240528194013.png]]
Above: An example of some of the tasks that vision language models can handle. Takes both images and text as input, and outputs text (or images, it seems).

## Overview of Open-Source Vision Language Models
- There are many on the hub, some are shown below
	- There are base models, and models tuned for chat that can be used in a conversational mode.
	- Some models have a feature called "Grounding," which reduces model hallucinations.
	- All models are trained on English, unless stated otherwise.

![[Pasted image 20240528194229.png]]
Above: [[LLaVA]] 1.6, [[DeepSeek]]-vl-7b-base, DeepSeek-VL-Chat, moondream2, [[CogVLM]]-base and [[CogVLM]]-Chat, Fuyu-8B, KOSMOS-2, [[Qwen]]-VL and Qwen-VL-Chat, [[Yi]]-VL-34B

## Finding the right Vision Language Model
- There are many ways to select the most appropriate model.
	- Vision Arena is a version of [[ChatBot Arena]] that's specific to VLMs, with its own leaderboard. The rankings are based on human preferences over head-to-head generations.
	- [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) is another one where models are ranked according to a variety of metrics.
- ==VLMEvalKit== is a toolkit to run benchmarks on vision language models, and powers the Open VLM Leaderboard. Another option is ==LMMS-Eval==, which provides a standard command line interface to evaluate HF models of your choice with datasets hosted on the HuggingFace Hub.

### MMMU
- [[Massive Multi-Discipline Multimodal Understanding|Massive Multi-Discipline Multimodal Understanding]] (MMMU) is the most comprehensive benchmark to evaluate VLMs on -- it contains 11.5k multimodal challenges that require college-level subject knowledge.

### MMBench
- [[MMBench]] is an evaluation benchmark consisting of 3000 single-choice questions from over 20 different skills, including OCR, object localization, and more.
- Also introduces an evaluation strategy called ==CircularEval==, where the answer choices of questions are shuffled in different combinations, and models are expected to give the right answer at every turn.

### Technical Details
- There are various ways to pretrain a language model; ==the main trick is to unify the text and image representations, and feed it to a text decoder for generation.==
	- The most common models consist of:
		- An image encoder
		- An embedding projector to align image and text representations (often a dense neural network)
		- A text decoder
- Training methods for models vary:
	- [[LLaVA]] consists of a [[CLIP]] image encoder, a multimodal projector, and a [[Vicuna]] text decoder. Authors fed a dataset of images/captions to GPT-4 and generated questions related to the caption+image. Authors then froze the image encoder/text decoder, and only trained the multimodal projector to align the image and text features by feeding the model images and generated questions, and comparing model output to ground truth captions. After projector finishes pretraining, they keep the image encoder frozen, unfreeze the text decoder, and train the projector *with* the decoder. ==This way of pretraining and finetuning is the most common way of training vision language models.== ![[Pasted image 20240528201206.png]]![[Pasted image 20240528201226.png]]

	- Another examples is KOSMOS-2, where authors chose to fully train the model end-to-end, which is computationally expensive compared to ==LLaVA-like pretraining described above.== Authors later did language-only finetuning to align the model. 
	- Most of the time, you don't need to pretrain a vision-language model, as you can use one of the existing ones or fine-tune them on your own use case.

## Using VLMs with `transformers`

Let's initialize the model and processor
```python
from transformers import LlavaNextProcess, LlavaNextForConditoinalGeneration
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretraiend(
	"llava-hf/llava-v1.6-mistral-7b-hf",
	torch_dtype=torch.float16,
	low_cpu_mem_usage=True
)
model.to(device)
```
Now let's pass the image and text prompt to the processor, and then pass the processed inputs to the `generate` (note: Each model uses its own prompt template; make sure to use the right one!)

```python
from PIL import Image
import requests

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"

# Using PIL to create an Image from our URL
image = Image.open(requests.get(url, stream=True).raw)

# This prompt template is specific to our model!
prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

# Process out inputs
inputs = processor(prompt, image, return_tensors="pt").to(device)

# Feed them to the model
output = model.generate(**inputs, max_new_tokens=100)
```










