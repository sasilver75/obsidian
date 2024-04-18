
![[Pasted image 20240331132326.png|450]]


In these two papers
- Progressive distillation: The process of using a teacher network to teach a student network
	- In the example of stable diffusion, we have a uNet that takes (eg) 64 steps to denoise an image from total noise -- why does it have to take so many? We use our trained uNet as a teacher, and create examples of (eg) {1-3}, {2-4}, {3-5}, etc. -- *two step* jumps! And we a student network to learn how to make "two" denoising steps at once! And then we do the same thing again with a *new* student network, using the *old* student as the *new* teacher! And we're able to collapse those 64 steps into something much smaller, like 4 steps!
- Distillation of Guided Diffusion Models:
	- We learned about classifier-free guided diffusion models last time, where we put in a prompt "cute puppy" into our CLIP text encoder, which spits out an embedding. We put that (ignoring VAE latents business) into our UNET, and *also* put the empty prompt into our CLIP text encoder, and concatenate the two embeddings together, such that out the other side, we get TWO things:
		- Image of cute puppy
		- Image of some arbitrary thing
	- We effectively do something like taking the weighted average of these two things, and use *that* for the next stage of our diffusion process.
		- We do the same student teacher distillation as before
		- But this time we also pass in the *guidance!*
		- We have the teacher model available for us, and we're doing actual [[Classifier-Free Guidance]] on our Diffusion, and we're doing it on a range of different guidance scales, and those now are becoming inputs to our student model
			- Our student model has additional inputs! It's getting:
				- The noise
				- The caption/prompt (as always)
				- Also getting the *guidance scale*, so it's learning ot find out how all of these things are handled by the teacher model.

Another paper came out about 3 hours ago too
![[Pasted image 20240331134728.png|450]]
With this algorithm, you can pass in an input image and pass in some text, saying (eg) "a bird spreading wings," and it will try to take the exact bird, the exact image, and try to keep it as similar as possible, but also match the prompt!
How cool!

The ability to take any photo of a person and change what the person is doing... is... societally very important, and really means that anyone can (Right now) generate believable photos that never actually existed.


... 



