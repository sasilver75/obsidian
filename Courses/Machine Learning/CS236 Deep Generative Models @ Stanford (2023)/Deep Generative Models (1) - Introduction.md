https://www.youtube.com/watch?v=XZ0PMRWXBEU

Instructor: Stefano Ermon, Associate Professor of CS @ Stanford
> "My research is centered on techniques for scalable and accurate inference in graphical models, statistical modeling of data, large-scale combinatorial optimization, and robust decision making under uncertainty, and is motivated by a range of applications, in particular ones in the emerging field of computational sustainability."

The goal of the class is to give you the foundations to understand the methods that are used in industry and academic papers works, and get up to speed. 

![[Pasted image 20240702002110.png|300]]
It will be a fairly mathematical class!

https://deepgenerativemodels.github.io/notes/index.html
Course site with notes

---

The challenge is to understand some complex, unstructured inputs 

From the perspective of a computer, an image is just a big matrix of numbers! How do we map this complex, high-dimensional object to some sort of useful representation for tasks that we care about, like figuring what objects are in an image, what they're made of, etc.
For NLP, a similar story -- we need to make sense of a series of characters. 

Understanding these objects is hard; it's not clear what understanding an image really means...

> "What I cannot create, I do not understand."
> - Richard Feynman's whiteboard, posthumously

He was talking about mathematical theorems, etc... but we can look at the contrapositive of this, in the context of generative modeling in AI:

> "What I understand, I can create"
> - Generative AI slogan?

If you claim to understand what an Apple is, you should be able to picture one in your hand, rotate it around, throw it up in the air, take a "bite" out of it and taste/feel/hear it.

So how do we build software that can generate images or text?
- This isn't really a new problem; people in computer graphics have been thinking about writing code that can generate images for a long time, for example!

Given a *high-level description* of a scene, the goal is to write some renderer that can produce an image that corresponds to that description.

![[Pasted image 20240701213239.png]]

If you can do this, then it seems you understand the concept of what a cube, cylinder, relative positioning, etc.

You can also imagine *inverting this process!*
- Given an image, what was the high-level description that produced this scene?

This gives you a way to think about Computer Vision  as being sort of like *inverse graphics!* Making progress in one might give traction on the other.


![[Pasted image 20240701213447.png]]
- Computer graphics rely on a lot of knowledge about physics, light transport, materials, etc.
- In this course, we'll try to use as little prior knowledge as possible, and leverage data (often collected on the public internet).

These statistical generative models are ==just going to be probability distributions p(x)==, eg over images x or sequences of text x.
- We'll build these models using Data (eg images of bedrooms), and the prior knowledge that we'll use will be things like the architecture, loss function, optimization strategy, etc.

![[Pasted image 20240701234420.png]]
It's generative because sampling from p(x) generates new images.

In some sense, what we're trying to do is build data simulators. Usually, data is the input. Here, in addition, our training input is data. Our inference input may include data and/or control signals.

![[Pasted image 20240701235148.png]]

A big improvement over the last 2-3 years was coming up with an idea of using score-based diffusion models
![[Pasted image 20240701235213.png]]
This was able to further push the SoTA at the time (Stanford PhD) student

![[Pasted image 20240701235258.png]]
Image generation conditional on text inputs

![[Pasted image 20240701235405.png]]
At the time of this lecture, [[DALL-E 3]] from OpenAI had just come out


![[Pasted image 20240701235440.png]]
You can often control models using different type of control signals; we've been thinking about how to colorize images, do impainting, do super-resolution, etc. These have become much easier to solve
![[Pasted image 20240701235810.png]]

![[Pasted image 20240702000044.png]]
WaveNet was a deep learning model for text-to-speech

Language generation has obvious been something that has been really exciting to people.
![[Pasted image 20240702000339.png]]
I think this is a GPT-2 generation...
![[Pasted image 20240702000412.png]]
This is with ChatGPT, the generation is clearly much better.

![[Pasted image 20240702000645.png]]
OpenAI Codex example, Microsoft Copilot

![[Pasted image 20240702000943.png]]

![[Pasted image 20240702001154.png]]
![[Pasted image 20240702001403.png]]

![[Pasted image 20240702001516.png]]

The course is designed to cover the core concepts in this space -- once you understand the different building blocks of what these models do... then you can not only understand how existing systems work, but maybe you can also design the next generation!
- The course is going to be rigorous and there's going to be quite a it of math
- The building blocks will be statistical modeling with probability distributions
- We'll talk a lot about how to represent distributions with many random variables, etc.

We're going to use data to fit models, but there are many different ways to fit models, different loss functions, how to measure similarity, etc.
- Diffusion models
- GANs
- LLMs, autoregressive models
Essentially boils down to different ways of comparing probability distributions, and trying to guide the model distribution to be in line with the data distribution.

We'll talk about inference, and how to generate samples efficiently.

![[Pasted image 20240702002059.png]]












