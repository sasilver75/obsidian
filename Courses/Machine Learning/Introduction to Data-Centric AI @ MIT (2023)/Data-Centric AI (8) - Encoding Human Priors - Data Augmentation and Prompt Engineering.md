https://www.youtube.com/watch?v=z44vZ_9av-M&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5&index=9

==This isn't a very useful lecture to me, it's all pretty basic content.==

Speaker (Sharon Zhou) did a PhD with Andrew Ng, taught a course on GANs on Coursera, has been fascinated with generative models and how far they've come.

How do we encode human knowledge of our world into data to improve our machine learning models? We'll cover two techniques:
- Data Augmentation: Focused on training the model
- Prompt Engineering: More appropriate at test time, a little newer

![[Pasted image 20240701113144.png|200]]

-----

ML models fail in interesting ways, once you walk off of golden paths
- A model trained on pictures of upright dogs will predict the "cat" label if we feed the same dog images rotated 90 degrees. We have a human prior that an imagine can be rotated in any direction and usually represent the same label, but our models have no such priors!
![[Pasted image 20240701113217.png|200]]

==Priors== are knowledge about the data, task, or world that we know before-hand
- We often take these for granted.

==Invariances==
- Changes to input data that don't change its output (flip it, rotate it, etc.)
- Dog can be in any orientation, and it's still a dog.
- Can we encode these human priors into our data?
	- This can be done in the data, loss function, or architecture of your model.
	- ((Does the model really learn these priors in a generalizable way, though? If I have a problem of classifying cats and dogs, and I just augment the dog image data with rotations, etc... does that have any positive transfer to understanding rotated cat images?))


![[Pasted image 20240701114525.png]]
As you load an image into a DataLoader in PyTorch, we can apply some transformation to our data. This is great for some transformations that are really efficient, but sometimes transformations are really expensive! There can be a lot, and it can be very computationally expensive -- sometimes we'll also do this image processing beforehand, as a result, and just load it regularly.



For some problems, collecting additional data is easy!
- More cat, dog images
For others, it's much more difficult
- Labeling medical images using board-certified pathologists

So Data Augmentation can come to the rescue


![[Pasted image 20240701115059.png]]
You can actually (in RGBA).... combine multiple images... and we give the model ("60% cat, 40% dog") as a label, rather than just "cat".  You don't have to one-hot encode your labels (think: multiclass classification), and you can still do a cross-entropy loss on it.


![[Pasted image 20240701115846.png]]


![[Pasted image 20240701115909.png]]

-----

Now that we've talked about encoding of human priors at training time, let's talk moer about how we can encode them at test-time!

![[Pasted image 20240701120230.png]]

Different models are going to respond differently (whether base vs instruct-tuned, or instruct-tuned vs rlhf'd, or even within categories) to different prompts.


















