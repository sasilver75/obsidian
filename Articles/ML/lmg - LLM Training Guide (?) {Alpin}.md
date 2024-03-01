---
tags:
  - article
---

Link: https://rentry.org/llm-training

-----

### Structure

1. The Basics
2. Training Basics
3. Fine-tuning
4. Low-Rank Adaptation (LoRA)
5. QLoRA
6. Training Hyperparameters
7. Interpreting the Learning Curves

------
# (1/7) The Basics

- The most common architecture used for language modeling is the [[Transformer]] architecture, introduced in the famous paper *Attention is all you need*. 
- Transformers allow us to train Large Language Models (LLMs) with incredible reasoning capabilities, while keeping the architecture simple enough for a novice in ML to get started on training/playing around with them.
- The most common language used for training and building Transformer models is Python; The most popular library in use is the [[HuggingFace]] `Transformer` library, which serves as the backbone of almost every LLM trainer today.
- ==LLMS are, in essence, a loss form of text compression==. We create tensors with random values and parameters, then feed a huge amount of text data so that they can learn the relationships between all the data and recognize patterns between them. All of these patterns are stored in the tensors we've randomly initialized as probabilities.
- A very high-level definition of LLMs would be "==compressing probability distribution of a language, such as English, into a collection of matrices==."

### Transformer Architecture
- It's always good practice to have an understanding of what you're working with... The best source of understanding is the *Attention is All You Need* paper. It introduced the Transformer architecture and is a profoundly important paper to read through.
	- HuggingFace blogposts and articles tend to provide easier-to-digest explanations than papers.

---
# (2/7) Training Basics

There are essentially 3 approaches to training LLMs:
1. Pre-Training
2. Fine-Tuning
3. LoRA/Q-LoRA

### Pre-Training
- Pre-training involves several steps:
	1. A massive dataset of text data is gathered (Many TB)
	2. A model architecture is chosen or created specifically for the task at hand.
	3. A tokenizer is selected or trained to appropriately handle the data.
	4. The dataset is then pre-processed using the tokenizer's vocabulary, converting the raw text into a format suitable for training the model.
	5. Now our data is ready to be used for the pre-training phase.
- Pre-training involves the model learning to predict the next word in a sentence or to fill in missing words by utilizing the vast amount of data available.
	- The pre-training phase thus typically employs a variant of the self-supervised learning technique. ==The model is presented with partially-masked input sequences, where certain tokens are intentionally hidden, and must be predicted using the surrounding context==.

---
# (3/7) Fine-tuning
- After the initial pre-training phase, where the model learns general language knowledge, ==fine-tuning allows us to specialize the models' capabilities and optimize its performance on a narrower, task-specific dataset.==
- The process of fine-tuning involves several key steps. 
	1. Firstly, a task-specific dataset is gathered, consisting of labeled examples relevant to the desired task. For exampe, if the task is *instruct-tuning*, a dataset of instruction-response pairs is gathered. The fine-tuning dataset size is significantly smaller than the sets typically used for pre-training.
	2. Next, the pre-trained model is then trained on the task-specific dataset, optimizing its parameters to minimize a task-specific loss function.
	3. To enhance the fine-tuning process, additional techniques can be employed, such as using a [[Learning Rate Schedule]], [[Regularization]] methods like [[Dropout]] or [[Weight Decay]] or [[Early Stopping]].


----
# Training Basics

### LoRA
Fine-tuning is computationally expensive, requiring hundreds of GBs of VRAM for training multi-billion parameter models.
To solve this specific problem, a new method was proposed: [[Low-Rank Adaptation]].
- ==LoRA can reduce the number of parameters by 10,000 times and the GPU memory requirements by over 3 times.==

### QLoRA
3x memory requirement reduction is still in the realm of unfeasible for the average consumer 
- Thankfully, a new LoRA training method was introduced: [[Quantized Low-Rank Adaptation]] (QLoRA). It provides near lossless quantization of language models, and then applies it to the LoRA training procedure.
- ==This results in massive reductions in memory requirement -- enabling the training/fine-tuning of models as large as 70 billion parameters on just 2x NVIDIA RTX 3090s, which would originally take more than 16x A100-80GB GPUs!==

### Training Compute
- Depending on the dataset size, the memory requirements will vary. You can refer to Eleuther's Transformer Math 101 blog post for easy-to-understand calculations.
- You'll want to fine-tune a model of AT LEAST the 7B class. Some popular options are [[LLaMA 2]] 7B and [[Mistral]] 7B, etc.
	- This size class typically requires memory in the 160-192GB range. 
	- Your options essentially boil down to:
		- Renting GPUs from cloud services (eg Runpod, VastAI, Lambdalabs, AWS Sagemaker)
		- Using Google's TPU Research Cloud
		- Know a guy who knows a guy

### Gathering a Dataset
- Dataset gathering is, without a doubt, the most important part of your fine-tuning journey.
	- Both quality and quantity matter, though quality is more important.
- First, think about what you want the fine-tuned model to do. Write stories? Role-play? Write emails? Waifu? 
- Datset structure:
	- Data Diversity: You don't want your models to *only* do one very specific task. You want to diversify your training samples, and include all kinds of scenarios so that your model can learn how to generate outputs for various types of input.
	- Dataset Size: Unlike LoRAs or Soft Prompts, you'll want a relatively large amount of data. As a rule of thumb, you should have ==at least 10MiB of data for your fine-tune== -- it's incredibly difficult to overtrain your model, so it's always a good idea to stack more data.
	- Dataset Quality:  ==The quality of your dataset is incredibly important==; you want your dataset to reflect how the model should turn out; If you feed it garbage, it'll spit out garbage.
### Processing the raw dataset
- You'll want to parse your dataset into a suitable format for pre-processing, depending on the format of your data:

1. HTML
	- You might have HTML files if you scraped your data from websites. In that case, you should pull your data out of the HTML elements. Rather than using RegEx, you should use a library like `Beutiful Soup` to help you with it.
2. CSV
	- The easiest way to parse this data is using the `pandas` python library, by reading the csv data into a pandas dataframe, selecting the column you need, casting it as a string type, and turning it into a string.
3. SQL
	- This one is a bit tougher; you can use a DB framework like PostgreSQL to parse the dataset into plaintext, but there are also many Python libraries for this purpose, like `sqlparse`

### Minimizing the noise
- The best language models are stochastic, which makes it difficult to predict their behavior, even when the input prompt remains the same.
- This can, on occasion, result in low-quality and undesirable outputs -- you want to make sure that your dataset is cleaned out of unwanted elements -- this is doubly important if your data source is synthetic, i.e. generated by GPT-4/3. 
- You might want to truncate or remove the mention of phrases like:
	- "As an AI language model...."
	- "harmful or offensive content..."
	- "...trained by OpenAI...", etc.
- [This script](https://huggingface.co/datasets/ehartford/wizard_vicuna_70k_unfiltered/blob/main/optional_clean.py) might be a goo filter for this task, or https://github.com/AlpinDale/gptslop.

### Starting the Training Run
- We'll use the [[Axolotl]] trainer for fine-tuning, since it's simple to use and has all the features we need.
- Axolotl takes all the options for training in a single `yaml` file.
- In our example, we're training the Mistral model using QLorRA, which should make it possible on a single 3090 GPU. To start the run, execute this command:

```
accelerate launch -m axolotl.cli.train examples/mistral/config.yml
```

Congrats, you just trained Mistral! To use a custom dataset, you might want to format it into a `JSONL` file. Axolotl takes many differnet formats; Then edit the `qlora.yml` file and point it to your dataset.


------
# (4/7) Low-Rank Adaptation (LoRA)
- LoRA is a training method designed to expedite ht training process while reducing memory consumption.
- ==By introducing pairs of rank-decomposition weight matrices (known as update matrices) to the existing weights, LoRA focuses solely on training these new added weights.==
- This approach offers several ==advantages==:
	1. ==Preservation of pretrained Weights==: LoRA maintains the frozen state of previously-trained weights, minimizing the risk of catastrophic forgetting. This ensures that the model retrains its existing knowledge while adapting to new data.
	2. ==Portability of trained weights==: The rank-decomposition matrices used in LoRA have significantly fewer parameters compared to the original model. This allows the trained LoRA weights to be *easily transferred* and utilized in other context, making them *highly portable*.
	3. ==Integration with Attention Layer==: LoRA matrices are typically incorporated into the attention layers of the original model. Additionally, the adaptation scale parameter allows control over the extent to which the model adjusts to new training data.
	4. Memory efficiency: LoRA's improved memory efficiency opens up the possibility of running fine-tune tasks on less than 3x the required compute for a native fine-tune!

### LoRA Hyperparameters
- **LoRA Rank**
	- This determines the number of rank decomposition matrices.
	- Rank decomposition is applied to weight matrices in order to reduce memory consumption and computational requirements. The original paper recommends a rank of 8 as the minimum amount.
	- Higher ranks lead to better results AND higher compute requirements -- the more complex your dataset, the higher your ranks will need to be!
	- To MATCH a full fine-tune, you can set rank to equal to the model's hidden size -- this isn't recommended, because it's a massive waste of resources.
- **LoRA Alpha**
	- This is the scaling factor for the LoRA, determining the extent to which the model is adapted towards new training data. The alpha value adjusts the contribution of the update matrices during the train process... Lower values give more weight to the original data and maintain the model's existing knowledge to a greater extent than higher values.
- **LoRA Target Modules**
	- Here, you can determine which specific weights and matrices are to be trained. 
	- The most basic ones to train are the eQuery Vectors (eg q_proj) and Value Vectors (v_proj) projection matrices. The names of these matrices will differ from model to model.
	- There are many more listed in the article.


-----
# (5/7) QLoRA
- QLoRA (Quantized Low Rank Adapters) is an efficient finetuning approach that reduces memory usage while maintaining high performance for large language models. ==It enables a finetuning of a 65B parameter model on a single 48GB GPU, while preserving full 16-bit fine-tuning task performance.==
- The key innovations of QLoRA include:
	- Backpropagation of gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA).
	- Use of a new datatype called a 4-bit NormalFloat (NF4), which optimally handles normally-distributed weights.
	- Double quantization to reduce the average memory footprint by quantizing the quantization constants.
	- Paged optimizers to effectively manage memory spikes during the finetunning process.



----
# (6/7) Training Hyperparameters
- Training hyperparameters play a crucial role in shaping the behavior and performance of your model. These hyperparameters are settings that guide the training process, determining how the model learns from the provided data. Selecting appropriate ones can significantly impact the model's convergence, generalization, and overall effectiveness.

### Batch Size and Epoch
- Stochastic gradient descent (SGD) is a learning algorithm with multiple hyperparameters for use -- two that often confuse a novice are the batch size and the number of epochs. They're both *integer* values that seemingly do the same thing.

- Stochastic Gradient Descent (SGD): It is an iterative learning algorithm that utilizes a training dataset to update a model gradually.
- Batch Size: The batch size is a hyperparameter in gradient descent that determines the number of training samples processed before updating the model's internal parameters. In other words, it specifies how many samples are used in each iteration to calculate the error and adjust the model.
- Number of Epochs: Controls the number of complete passes through the training data; Each epoch involves processing the entire dataset once.

### Stochastic Gradient Descent (SGD)
- A search process where the algorithm learns to improve the model.
- The algorithm works iteratively,  . This error is then utilized to update the internal model parameters. This error is then utilized to update the internal model parameters.

### Sample
- The batch size is a hyperparam that determines how many samples are processed before updating the model's internal parameters.
- After processing a batch, the predictions are compared to expected outputs, and an error is calculated. This error is then used to improve the model by adjusting its parameters, moving in the direction of the error gradient.
- Several types of gradient descent algorithms are used, based on batch size:
	- ==Batch Gradient Descent==: When the batch size is equal to the total number of training samples.
	- ==Stochastic Gradient Descent==: When the batch size is set to one sample. This approach introduces more randomness into the learning process.
	- ==Mini-Batch Gradient Descent==: When the batch size is greater than one and less than the total size of the training dataset. Strikes a balance between the efficiency of batch gradient descent and the stochasticity of stochastic gradient descent.

What happens when the dataset size doesn't divide evenly by the batch size?
- This happens often; it simply means that the final batch size has fewer samples than the other ones.


### Epoch
- The number of epochs is a hyperparameter that determines ==how many times the learning algorithm will operate over the entire dataset.==
- One (1) epoch signifies that each sample in the training dataset has been used once to update the model's internal parameters. An epoch consists of one or more batches.
- You can visualize epochs as the outer loops, and iterating over batches as the inner loop.
- To assess the model's performance over epochs, it's common to create line plots, also known as learning curves. These plots display epochs on the x-axis as time and the model's error or skill on the y-axis.

### Batch vs Epoch
- The batch size is the number of samples processed before the model is updated.
- The number of epochs is the number of complete passes through the training dataset.
- The size of a batch is >= 1 and <= the number of samples.
- The number of epochs can be set to any positive integer.

- Keep in mind taht larger batch sizes result in higher GPU memory usage! We'll be using ==Gradient Accumulation Steps== to update this!


### Learning Rate
- As discussed in the section for Batch and Epoch, in machine learning, we often use an optimization method called stochastic gradient descent (SGD) to train our models. One important factor in SGD is the learning rate, which ==determines how much the model should change in response to the estimated error during each update of the weights==.
- Think of the learning rate as a knob that controls the size of steps taken to improve the model. 
	- If the learning rate is too small, the model may take a long time to learn or get stuck in a suboptimal solution.
	- If the learning rate is too large, the model may learn too quickly and end up with unstable or less accurate results.
- Choosing the right learning rate is crucial for successful training. It's like finding the goldilocks zone -- not too small, too large -- just right.

### Learning Rate and Gradient Descent
- Stochastic gradient descent estimates the error gradient for the current state of the model using examples from the training dataset, and then updates the weights of the model using backpropagation.
- Specifically, the learning rate is as configurable hyperparameter used in training that has a very small positive value, often in the range between 0 and 1.
- Larger learning rates result in rapid changes and require fewer training epochs; The learning rate is ==perhaps the most important hyperparameter -- if you have time to tune only one hyperparameter, tune the learning rate==.

### Configuring the Learning Rate
- Start with a reasonable range: Begin by considering a range of learning rate values commonly used in similar tasks. For example, find the learning rate used for the pre-trained model that you're fine-tuning, and base it off that.
- Observe the training progress; Run the process with the chosen learning rate and monitor the progress during training.
- Too slow? If the learning rate is too small, you might notice that the model's training progress is slow and takes a long time to make noticeable improvement. Consider improving the learning rate.
- Too fast? If the learning rate is too large, the model may learn too quickly, leading to unstable results. Signs of a too-high learning rate 
- Iterative adjustment: Based on what you see in steps 3 and 4, iteratively adjust the learning rate and r-run the training process. ==Gradually narrow down the range of learning rates and re-run the training process.==

==A general-purpose formula for calculating the learning rate is:==

`base_lr  * sqrt(supervised_tokens_in_batch / pretrained_batch_size)`
Where base_lr refers to the pre-trained model's learning rate.
For reference, the base learning rate for Llama-2 is 3e-4

### ==Gradient Accumulation==
- Higher batch sizes result in high memory consumption. [[Gradient Accumulation]] aims to fix this.
- ==Gradient accumulation is a mechanism to split the batch of samples -- used for training your model -- into several mini-batches of samples that will be run sequentially==.
![[Pasted image 20240201000937.png]]


### Backpropagation
- Gradients help us understand how the loss value changes with respect to each model parameter. Think of gradients as arrows that show us the direction and magnitude of the change in the loss as we tweak the parameters.
- Once we have the gradients, we can use them to update the model's parameters and make them better. We choose an ==optimizer==, which is like a special algorithm responsible for guiding these parameter updates. ==The optimizer takes into accounts the gradients, as well as other factors like the learning rate (how big the updates should be) and momentums (which help with the speed and stability of learning).==
- To simplify, let's consider a popular optimization algorithm called SGD
	- `V = V - (lr * grad)`
	- V is any parameter, lr is the learning rate, and grad is the gradients we calculated earlier.


### Iteration
- Let's say we are accumulating gradients over 5 steps. 
	- In the first 4 steps, we calculate gradients and store/retain them, but we don't update any variables. 
	- Then, in the fifth step, we combine the accumulated gradients from the previous steps *and* the current step in order to calculate and assign the variable updates.
- We can simply sum these gradients, and, with the gradients combined, we can compute the variable updates and assign them accordingly.
	- `V[t] = V[t-1] - lr * (Sum of accumulated gradients)`

### Configuring the number of gradient accumulation steps
- As we've discussed, you would want to use gradient accumulation steps to achieve an effective batch size that is close to or larger than the desired batch size.
- For example, if the desired batch size is 32 samples, but you have limited VRAM that can only handle batch sizes of 8, then you can set the gradient accumulation steps to 4. This means that you accumulate gradients over 4 steps before performing the update, effectively simulating a batch size of 32.




---
# (7/7) Interpreting the Learning Curves

Learning curves are one of the most common tools for algorithms that learn incrementally from a training datasets. The model is evaluated using a validation split, and a plot is created for the loss function, measuring how different the model's current output is compared to the expected one.


# Overview
A learning curve can be likened to a graph that presents the relationship between time or experience (x-axis) and the progress or improvement in learning (y-axis), using a more technical explanation.

There are two types of learning curves that are commonly used:
- ==Train Learning Curve==: This curve is derived from the training dataset and gives us an idea of how well the model is learning during training.
- ==Validation Learning Curve==: This curve is created using a separate validation dataset. It helps us gauge how well the model is generalizing to new data.

Sometimes, we might want to track multiple metrics for evaluation.
For example, in classification problems, we might optimize the model based on cross-entropy loss and evaluate its performance using classification accuracy.

Two types of learning curves:
- ==Optimization Learning Curves==:
	- These curves are calculated based on the metric (eg loss) used to optimize the model's parameters.
- ==Performance Learning Curves==:
	- These curves are derived from the metric (e.g. accuracy) used to evaluate and select the model.

### Model Behavior Diagnostics
- The shape and dynamics of a learning curve can be used to diagnose the behavior of a model and in turn perhaps suggest at the type of configuration changes that may be made to improve learning and/or performance.
- There are 3 common dynamics that you're likely to observe in learning curveS:
	1. Underfit
	2. Overfit
	3. Well-fit

### Underfit learning curves
- This refers to a model that is unable to learn the training dataset.
- You can identify an underfit model from the curve of the training loss only.
- It may show a flatline or noisy values of relatively high loss, indicating that the model was unable to learn the training dataset at all.

![[Pasted image 20240201004005.png]]
- The training loss remains flat regardless of training
- The training loss continues to decrease until the end of training


### Overfit learning curves
- This refers to a model that's learned the training dataset *too well,* leading to a memorization of the data, rather than generalization.
	- This would include the statistical noise or random or random fluctuations present in the training dataset, which we *wouldn't* like to learn.
- The problem with overfitting is that the more specialized the model becomes to the training data, the less well it's able to generalize to new data, resulting in an increase in generalization error.
- This often happens when the model has more capacity than is necessary for the required problem, and, in turn, too much flexibility. It can also occur if the model is trained for too long.
- A plot of learning curves show overfitting if:
	- The plot of training loss continues to decrease with experience.
	- The plot of validation loss decreases to a point, and begins increasing again.

![[Pasted image 20240201004336.png]]
The inflection point in validation loss may be the point at which training could be halted as experience after that point shows the dynamics of overfitting.

### Well-fit learning curves
- This would be your goal during training - a curve between an overfit and underfit model.
- A good fit is usually identified by a training and validation loss that decrease to a point of stability with a minimal gap between the two final loss values.
- The loss of the model will always be lower on the training dataset than the validation dataset; we should expect *some gap* between the train and validation loss curves. This is referred to as the ==generalization gap==.
- The example plot would show a well-fit model:
![[Pasted image 20240201004641.png]]


-------





























