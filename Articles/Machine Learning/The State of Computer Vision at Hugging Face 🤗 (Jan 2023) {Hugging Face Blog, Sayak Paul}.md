#article 
Link: https://huggingface.co/blog/cv_state

This wasn't that good of an article, I wouldn't recommend rereading it.

----

Last year (2022), we began focusing efforts on CV at HF; we started having [[Vision Transformer]]s (ViT) in our `Transformers` library, and now we have ~8 core vision tasks, >3000 models, and over 100 datasets on HF hub!

Here's a list of things we'll cover:
1. Supported vision tasks and Pipelines
2. Training your own vision models
3. Integration with `timm`
4. Diffusers
5. Support for third-party libraries
6. Deployment
More!

---

# Enabling the Community: One task at a time


Today (Jan 2023), we support 8 core vision tasks, providing many model checkpoints:

- [[Image Segmentation]]
- [[Image Classification]]
- [[Object Detection]]
- [[Video Classification]]
- [[Depth Estimation]]
- Image-to-image synthesis
- Unconditional image generation
- Conditional image generation

Each of these tasks comes with at least 10 model checkpoints on the hub to explore!

More tasks at the intersection of language and vision:
- Image-to-text
- Text-to-image
- Document question-answering
- Visual question-answering

These tasks entail state of the art Transformer models (ViT, Swin, DETR), but also pure convolution architectures (ConvNeXt, ResNet, RegNet).


# Support for Pipelines 🧪
- We developed ==Pipelines== as a way to easily perform inference ona given input with respect to a task:
```python
from transformers import pipeline

# Create a Pipeline using this (lowercased?) functino
depth_estimator = pipeline(task="depth-estimation", model"Intel/dpt-large")

# Invoke our Pipeline on some input data to get a result
output = depth_estimator("http://images.cocodataset.org/val2017/00000039769.jpg")

# A tensor with the values for the depth expressed in meters for each pixel
output["depth"]

```
![[Pasted image 20240415182708.png|300]]

It even works for visual question-answering!

```python
from transformers import pipeline

oracle = pipeline(model="...")
image_url = "https://hugginface.co/datasets/mishig/sample_images/.../tiger.jpg"

oracle(question="What's hte animal doing?", image=image_url, top_k=1)

```

# Training your own models 🦮
- While being able to use a model off-the shelf like above is a great way to get started, fine-tuning is where the community benefits the most.
- Transformers provides a ==Trainer API== for everything related to training! Currently, `Trainer` seamlessly supports the following tasks:
	- Image classification
	- Image segmentation
	- Video classification
	- Object detection
	- Depth Estimation

# Integrations with Datasets 💽
- Datasets provide easy access to thousands of datasets of different modalities that can be loaded in only a few lines of code, using our ==Datasets== API.

```python
from datasets import load_dataset  # Nice

dataset = load_dataset("scene_parse_150")
```

Besides these datasets, we provide integration support with augmentation libraries like albumentations and Kornia.


# timm 🤝
- The ==timm== library, also known as ==pytorch-image-models== is an open-source colleciton of SoTA PyTorch image models, pretrained weights, and utility scripts for training, inference, and validation.
- We have over 200 models from `timm` on the hub, and more on the way.

# Diffusers 💣
- The ==Diffusers== library from HF provides pre-trained vision and audio diffusion models, serving as  modular toolbox for both inference and training.
```python
from diffusers import DiffusionPipeline

generator = DiffusionPipeline.frmo_pretrained("CompVis/stable-diffusion-v1-4")
generator.to("cuda")

image = generator("An image of a squirrel in Picasso style").images[0]

```


# Spaces for computer vision demos
- With ==Spaces==, we can easily demonstrate our ML models by integrating with [[Gradio]], [[Streamlit]], and Docker, empoewring practitioners to have a good amount of flexibility when showcasing models.

# AutoTrain
- AutoTrain Provides a "no code" solution to train SoTA ML models for text classification, text summarization, NER, and more -- For CV, we currently support Image Classification, but one should expect more task coverage soon.

# The Technical Philosophy

Even though Transformers started with NLP, we support multiple modaliities/methods today:
1. Vision
2. Audio
3. Vision-Language
4. Reinforcement Learning

For all of these, the corresponding models from `Transformers` enjoy some common benefits:
- Easy model downloads with `from_pretrained()`
- Easy model upload with `push_to_hub()`
- Support for loading huge checkpoints with efficient checkpoint sharding techniques
- Optimization support (eg with Optimum)
- Initialization from model configurations

We have preprocessors that take care of preparing data for vision models, working hard to make to UX of using a vision model easy and familiar:

```python
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset

# Using datasets to import a dataset from the Hub
dataset = load_dataset("huggingface/cats-iamge")
image = daataset["test"]["image"][0]

# Creating an image processor and our model
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-...")
model = ViTForImageClassification.frmo_pretrained("google/vit-base-...")

# Preprocess (?) an image, returning a new input image
inputs = image_processor(image, return_tensors="pt")

# Run prediction on that image, retrieve logits
with torch.no_grad():
	logits = model(**inputs).logits

# model predits one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
# Egyptian cat 😺


```

Cool

# Zero-Shot Models for Vision
- There have been a surge of models that reformulate core vision tasks like segmentation and detection in interesting ways, producing even more flexibility.
- [[CLIP]] enables zero-shot image classification with prompts.
- OWL-ViT allows for language-conditioned zero-shot object detection and image-conditioned one-shot object detection.... meaning you can detect objects even if the underlying model didn't learn to detect them during training!
- CLIPSeg supports language-conditioned zero-shot image segmentation and image-conditioned on-shot image segmentation.

