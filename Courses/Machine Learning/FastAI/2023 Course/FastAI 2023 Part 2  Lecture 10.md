
Recap of the previous lecture:
![[Pasted image 20231210120513.png]]
- The basic idea is that we start with (eg) a collection of handwritten digits. 
- We add to it some gaussian random noise. The 7 plus the noise together equals the noisy seven.
- We present the noisy seven to our network, and have it try to predict the noise in the image.
- It then compares the prediction to the actual noise, and gets the loss that it uses to update the weights in the UNet.
- That's the basic loop!
- To make it easier for the UNet, we can pass in an *embedding* of the number 7 (eg a one-hot encoded vector); If we do this, then later on we can generate *specific* digits by saying "I want a number 7" or "I want a number 5"
- We skipped over the VAE piece, which is just a computational shortcut that makes it faster; We use the encoding part of an autoencoder to preprocess our images of fuzzy 7s into a latent space first.

We want to handle things that are more interesting than the number 7, though!

![[Pasted image 20231210120733.png]]
- We want to handle things like "A graceful swan" and "A scene from hitchcock"!
- We do this by creating separate embeddings of both the pictures and of their labels, and specifically, we try to make the embeddings "similar," in the sense that the image of the graceful swan and the *text* of the graceful swan both embed to the ~same space. IE, they both have the same semantic meaning.
	- The way we do this (in CLIP) is to just download from the internet lots of images, find their *alt* tags, and then for each one, we have the image and its alt tag, and we build two models:
		- An image encoder that turns each image into its feature vector
		- A text encoder that turns each piece of text into a bunch of features
		- A loss function that says "The graceful swan text" vector should be as similar as possible to the "Picture of a graceful swan" ... We add up all the green ones, and subtract all the red ones in the picture above, and that's the [[Contrastive Loss]].
- Now we have a text encoder where we can say "A graceful swan," and it will spit out some embeddings, and those embeddings can be passed into the UNet during training.

So we haven't been doing any of that training ourselves, except some fine tuning (because it's expensive); instead, we take some pretrained models and do some inference!
- We put in an example of something that we have an embedding for, along with some random noise into the UNet, and it spits out an idea of which bits of noise you could remove to leave behind some pictures of the number 3!
- Initially it's going to do a bad job of that, with the first pass; so we subtract just a little bit of the noise *several times* by just looping the output back into the input.

![[Pasted image 20231210121704.png]]
Above: This is an example of smiling picture of Jeremy Howard.

Paper Explanation:
*Progressive Distillation for Fast Sampling of Diffusion Models* (Samilans, Ho @ Google) and *On Distillation of Guided Diffusion Models (Meng)*
- These papers have taken the required number of steps from 60 steps or so down to just ~4 steps!
- In the Progressive Distillation paper above...

So how are we going to get down from 60 inference steps down to 4 steps?
- We're going to do a process called [[Distillation]].  This is a process that's pretty common in deep learning.
	- The basic idea is that you take something called a Teacher Network that already knows how to do something, but it's slow and big. The Teacher Network is used by the Student Network to try to do the same thing *faster* or with *less memory*! 
	- The way we do this is actually pretty straightforward!
		- Question: Why is it even taking us 60 steps to go from static to 60? Or from 36 to 54, above? That seems like it should take just one step, right? The reason is... just a side effect of the math of how this thing was originally formulated (the diffusion process)
		- In the paper, what if we train a *new model* where the model takes as input (eg) the 36 image, and puts it through some *other* UNet B... then *that* spits out some result, and what we do is take that result and compare it to the 54 image, which is what we actually want!
			- Because given that we've done the full 0-54 iterations using the UNet A, we have, for each image, both the output at that iteration, and the output at the 54'th iteration, which is what we're actually trying to work towards.
			- So we can try to take the output of 36 as input, and try to predict 54 as the output, using UNet B, using something like MeanSquaredError (MSE).
		- If we keep doing this for lots and lots of images, this UNet B is going to learn to take these incomplete images and turn them into complete images! That's exactly what this paper does -- "Now that we've got all these examples showing what step 36 should turn into at step 54, let's just turn that into a model. And that makes sense, because really you'd expect a human to be able to do that."
		- ![[Pasted image 20231210125822.png]]
		- There are some practical tweaks and details to this:
			- They initially take their Teacher Model (UNet A, a complete stable diffusion model), and we put in our random noise... and we put it through *two timesteps,* and then we train our UNet B to go from {noise} to {timestep 2}. 
			- Then we take that student model, and we treat it as the new teacher! We now take our noise, and run it through the *student model* twice, and get a {timestep 2} out the other end.
			- Then they try to create a *new* student, which is a copy of the previous student that tries to learn how to predict that output from the previous student model!
			- And they continue doing this, again and again, each time basically doubling the amount of work that a student is able to do. This is basically what they're doing! 
![[Pasted image 20231210130025.png]]

Let's talk about the other paper -- *On Distillation of Guided Diffusion Models*
- What's up with this paper?
- We want to be able to do guidance! Last time, we used something called [[Classifier-Free Guidance]] Diffusion (CFGD) models. In this one, if we want a cute puppy, we put in the prompt "Cute puppy" into our CLIP text encoder, and it spits out an embedding! We put that embedding (ignoring the VAE Latent business) into our UNet; but we also put the *empty* prompt into our CLIP text encoder! We concatenate these two things together so that we get back *two* things: 
	- We get back the image of the puppy, and we get back an image of the arbitrary thing that resulted from the empty prompt.
	- We effectively then do something like taking the weighted average of those two images together and combine them; Then we use that combined image for the next stage of the diffusion process.
	- ![[Pasted image 20231210134217.png]]
- This paper basically says: "This process is pretty awkward; we have to end up trainign two images instead of one; For different types of guided diffusion, we have to do it all multiple times -- how do we skip it?"
- What we do is exactly the same student-teacher Distillation of our model, but this time we pass in (in addition) the guidance! So we've got the entire stable diffusion model (the teacher model) available for us, and we're doing actual CFGD (Classifier-Free Guided Diffusion) to create our cute puppy pictures, and we do it for a range of different guidance scales (2, 7.5, etc.)... And those are now becoming *inputs* to our student model! So our student model has additional inputs:
	- It's getting the noise (As always)
	- The caption/prompt (as always)
	- But it's also getting the guidance scale
- So the student is learning to find out how all of these things are handled by the teacher model -- what does it do? It's just like before, but now it's learning to use the Classifier-Free Guided Diffusion as well.

Paper: *Imagic: Text-Based Real Image Editing*
- ![[Pasted image 20231210150026.png]]
- ![[Pasted image 20231210150050.png]]
- So we actually already know how something like this might work!
- Here's how it works, at a low level of detail:
- ![[Pasted image 20231210150122.png]]
- We start with a fully pretrained, ready-to-go generative model, like a stable diffusion model. In the paper they use one called Imagen, but it really doesn't matter what the model is.
- We take a picture of a bird spreading its wings (our target), and we create an embedding from that eg using our CLIP encoder as usual
- And we pass that image through our pre-trained diffusion model, and we then see what it creates (A). It doesn't create something that's like our bird with the wings spread; instead, it creates... some kind of bird, or something. 
- So then we fine-tune this embedding to try to make the diffusion model output something that's as similar as possible to the input image of the bird with the wings spread (see teh image where they're moving the embedding a little bit). Now they lock that in place, and say "Now let's fine-tune the entire diffusion model end-to-end, including the VAE... and now the optimized embedding we created, we store it in place; that's now frozen. And we try to make it so that the diffusion model spits out our bird, as closely as possible."
- Now we have something that takes this embedding, goes through the fine-tuned model, and pits out our bird.
- Now finally, the original target embedding we actually wanted was a photo of a bird spreading its wings; we ended up with a slightly different embedding. We create a weighted average of the two embedding in the "interpolate" step, and we pass it through the fine-tuned diffusion model, and we're done!

















