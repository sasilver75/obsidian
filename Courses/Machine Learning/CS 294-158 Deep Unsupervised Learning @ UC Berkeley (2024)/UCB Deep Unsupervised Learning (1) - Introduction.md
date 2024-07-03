https://youtu.be/tFR6Likf4VI?si=kNKYZ-tM4PIsN7ce

![[Pasted image 20240630215850.png]]

-----

![[Pasted image 20240630221124.png]]
The label-free part really matters -- it's the original motivation for unsupervised learning, because labeling is time-consuming, and it's nice if you don't have to do it. But there are additional motivations beyond it being label-free, which we'll learn about soon!

Two sub-areas of deep unsupervised learning:
- Generative models
- Self-supervised Learning

Why care?
- In a lot of ML (supervised), there's a very clear, deterministic solution, in that you've got an image, and a clear label has to be assigned to it.
- Whereas in generative models, we're trying to model pretty complex distributions using a NN, which is hard to do! That's why we dedicate 5 3-hour lectures to how to approximate complex distributions with neural networks!

Why care?
![[Pasted image 20240630221549.png]]
Jeff worked on something for 40 years that he thought was going to be correct... and after 40 years, he and his students had success with AlexNet.
- Basically, there's so much data that we need, and unsupervised learning means that we can take advantage of unlabeled data.

![[Pasted image 20240630222709.png]]
In 2016, all of the excitement at NeurIPS was about Reinforcement Learning. He wanted people to pay more attention to unsupervised learning -- we need a lot of data, and unsupervised learning is the way to get a lot of it.
- Funnily, ChatGPT is exactly this cake! (SLS, SFT, RLHF)

![[Pasted image 20240630223523.png]]
- [[Kolmogorov Complexity]]: The size of a dataset measured in KC is the size of the shortest computer program that can represent that dataset.
	- The simplest way to do it (might not be the winning size) is just to store the entire dataset, and have your program print it out. Rarely gonna be the one that wins, but if your dataset is truly random, that might be all that you can do.
	- Think about your NN as the program. So what's the most compact NN that can regenerate your data? That NN is the one that best understands your data (Smallest NN doesn't necessarily mean the smallest number of parameters; it could mean more, lower-precision parameters).
- [[Solmonoff Induction]]: In principle, shortest code-length should allow for optimal inference. 
- Extensible to optimal action making agents (AIXI): What's the smallest agent I can build to solve a given problem?

[[Ilya Sutskever]]: If we pretrain on distribution D1 and finetune on D2, then if D1 and D2 are related, then compressing D2 conditioned on already knowing D1 should be more efficient that compressing D2 outright.
Not super mathematically precise, but an intuitive argument for why unsupervised learning should help, when later we only care about supervised learning.



Aside from theoretical interest, there are many powerful applications
- Generate Novel Data
- Conditional Synthesis (WaveNet, GAN-pix2pix)
- Compression (Interestingly, this hasn't been commercialized yet; how good you can compress a dataset depends on how good a model you have of p(X), and the bound of that is the Entropy of that distribution)
	- If compute chips become larger and cheaper, maybe we can have a massive NN that can decode the movies that you want to watch coming over the wire, because it can decode efficiently from an incredibly small latent representation being shot over the wire.
- Improving downstream tasks like [[Self-Supervised Learning]]
- Flexible building blocks that can be used in many systems.


![[Pasted image 20240630224045.png]]
Deep Belief Nets showed that we can generate images that look like the images in our dataset. This was pretty surprising at the time.


![[Pasted image 20240630224223.png]]
The [[Variational Autoencoder]] (VAE) was also a big step forward in 2013


![[Pasted image 20240630224236.png]]
[[Generative Adversarial Network]]s (GANs) in 2014 were a breakthrough
Their low resolution was just a compute limitation at the time.
- Ian was having drinks with friends in a bar in Montreal, and when they're an AI student having drinks, they talked about AI -- "How is it possible that NNs can't create proper images? Even a NN can't see these images aren't sharp...wait, should we use another critic NN in a feedback loop to give feedback back to the generator?"

![[Pasted image 20240630224335.png]]
DCGANs in 2015 ([[Alec Radford]]) generated the first really high quality images. He dropped out to futz with AI, then OpenAI recruited him.
![[Pasted image 20240630224605.png]]
Some more DCGAN images; humans tend to be very sensitive to artifacts in faces; we still don't see these as realistic.

In 2017, SuperResolution GANs, CycleGANs, and more me.
In 2018, [[DeepMind]] did the first very large GAN training, with BigGAN.

![[Pasted image 20240630224758.png]]
In 2018, StyleGAN came out of NVIDIA, and these faces becaem pretty much indistinguishable from real faces.

![[Pasted image 20240630224836.png]]
But then, [[Diffusion Model]]s came on the scene in 2020; the challenge people ran into with GANs was that even though everything was unrealistic, it wasn't covering the entire distribution that the data had! It focused on particular *modes* of the distribution, without covering the entire distribution. Can we have realism and good coverage of the input distribution?
- These are the models powering almost all of image generation today.

![[Pasted image 20240630225048.png]]
If an AI model understands this prompt, it means that it really has some understanding of the world -- surely this text wasn't present in the input data.
![[Pasted image 20240630225107.png]]
Again, this surely wasn't an image/text caption that was found in our training set, right?

![[Pasted image 20240630225131.png]]
![[Pasted image 20240630225223.png]]

Since then, OpenAI's [[DALL-E]] and Google's [[Imagen]] came out
After that, Stability AI's [[Stable Diffusion]], MidJourney came out, etc.


![[Pasted image 20240630225357.png]]
We can generate audio too!

![[Pasted image 20240630231544.png]]
Or generate audio from text!
If we do autoregressive prediction, we can generate the samples one-by-one; instead of predicting things in the raw audio space like in WaveNet, first we'll tokenize everything into the same space (text and audio), and then we can just concatenate (text+audio), and just generate it 🎶

![[Pasted image 20240630233922.png]]
Tacotron was a landmark work saying we can take in text, do some processing, and et out these spectrograms to get some raw audio out.
- These days, elevenlabs has been making text-to-speech better internally.

![[Pasted image 20240630234047.png]]
Video is a new domain that's just starting to work.

![[Pasted image 20240630234228.png]]
Again, we tokenize images into image tokens
Then we have text tokens and image tokens mapped to the same space, and generate :)
A dog wearing a VR headset eating pizza isn't in the data distribution, but we can generate it!

![[Pasted image 20240630234342.png]]
Getting robots to plan; to imagine what would happen if we did a simple action. We can roll that out in simulation, instead of the real world, to help a robot with planning! We unsupervised learning -based simulation on the inner loop of RL planning!

![[Pasted image 20240630234430.png]]
From Andrej's "The Unreasonable Effectiveness of RNNs", where he did character-level generation using an RNN. 

![[Pasted image 20240630234514.png]]
We can also use them to generate things like Math

![[Pasted image 20240630234533.png]]
GPT-2, OpenAI's next attempt at scaling things up
- The main innovation here is that with the right prompting, we can get unsupervised models to do the things we want them to do. So we can train on giant distributions of text, and then using prompting to guide behavior.

![[Pasted image 20240630235032.png]]

![[Pasted image 20240630235022.png]]

A lot of the power of these models come from the fact that they aren't supervised -- we don't need labels to train them.

![[Pasted image 20240701000741.png]]









