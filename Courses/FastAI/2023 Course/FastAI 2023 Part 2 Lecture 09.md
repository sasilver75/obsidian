
We're going to be using the Hugging Face `Diffusers` library today to introduce [[Stable Diffusion]], the highest-quality open source ***text to image model*** as of this course's release!

---
Aside: The `Diffusers` library from Hugging Face
- It's the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules
- Three core components:
	- State of the art "diffusion pipelines" that can be run in inference with just a few lines of code.
	- Interchangeable noise "schedulers" for different diffusion speeds and output quality.
	- Pretrained "models" that can be used as building blocks, and combined with schedulers to create new end-to-end diffusion systems.

---


We'll see that understanding state-of-the-art generative models requires a deep understanding of many of the fundamental building blocks in machine learning! 

We're going to see what [[Stable Diffusion]] can do, and get a glimpse of its main components!

Let's get an understanding about how this stuff might work...

Stable diffusion is usually explained as some sort of mathematical explanation, but we're going to try something different...and conceptually much simpler!


We have a bunch of (say) 28x28 images of hand written digits. We want to train a model that is able to generate more handwritten digits!
Our strategy is this:
1. We in sequence apply, for each handwritten digit, a number of noise-adding filters (eg 10%, 20%, 30%, ...) that just add random noise/fuzz to the image. So now we have a number of pictures of a 7, with varying degrees of fuziness!
2. We know that our (fuzzy 7 image) is actually the elementwise sum of (7 image) plus (fuzziness that was added). 
3. We train a neural net to be able to PREDICT the fuzziness that was added. We can do this because we know what the non-fuzzy version of the (7 image) looked like. Just use MSE to get the pointwise difference, or something, to train it.
4. Now that we have an neural net that, given an image, is able to predict the aspects (in this case, fuzzy noise) of the image that ARE NOT part of the handwritten digit... we're actually able to generate handwritten digits, believe it or not!
5. We pass it a VERY noisy image (pure noise). We pass it to the NN, and it will spit out information saying which part of it that it thinks is noise! This implicitly will leave behind (by subtraction) the bits that look most like a digit!
	- "You know what, if you left behind this bit, that bit, and that bit, it'd look more like a digit."
6. We subtract the predicted noise bits (times some constant), and we have some new image that looks... a little more like a digit, which is what we hoped for.
7. Then we can just do it again!
![[Pasted image 20231209120631.png]]
You can see why we're doing this multiple times...

In practice, we use a particular type of Neural Net for this; a type of network designed for medical imaging called the [[U-Net]]. 

This is the first component of stable diffusion!

[[U-Net]]
- The input is a somewhat noisy image (it could be anywhere from all noise to not noisy at all)
- The output is the noise (such that if we subtract the output from the input, we end up with an approximation of the unnoisy image)

But we want to generate images of great paintings and stuff!

We (at the moment) use something like a 512x512x3 channel RGB image.

When you think about it, storying the exact pixel value of every single pixel might not be the most efficient way to store it (in theory)... there are faster, more concise (possibly less accurate) ways of storing the information in an image.

We know this is true, because a JPEG picture is far fewer bytes than you'd get if you multiplied it's height x width x channels. So we know that it's possible to compress pictures in some way.

An interesting way to compress images:
- Take the image, and put it through a convolutional layer of Stride=2, with 6 channels.
- That would turn our 512x512x3 image into a 256x256x6 image.
- Then let's put it through another Stride=2 convolution, to get 128x128x12
- And another Stride=2 convolution, to get 64x64x24.
- And now let's put that through a few ResNet blocks to squish down the number of channels as much as we can... Say it's 64x64x4
- ![[Pasted image 20231209124206.png]]
Let's say that we now have 64x64x4=16,384 pixels
We've compressed from 786,432 -> 16,384, which is a ~48x decrease
That's no use, though, if we've lost our image! But can we get the image back again? Sure, why not!
We basically create an "inverse convolution" of some sort, that does the exact opposite!
- We'll take our 64x64x4 image...
- Put it through an inverse convolution back to 128x128x12
- Another inverse convolution back to 256x256x6
- Back to 512x512x3![[Pasted image 20231209124450.png]]

This whole thing is a single neural network that we could feed:
- An image in the frontend, and an image is received in the backend.
- The image coming out the back is initially going to be shitty -- we need a loss function!
- Let's just determine the MSE between the input and output image!

What will that do to the model?
- It will try to make sure that the image that comes out the other end of the NN is exactly the same as the image that goes into it!

This seems really boring, right? What's the point of a model that only learns to give you back exactly what came in?

This kind of model is called an [[AutoEncoder]]; it gives you make what you gave it.
- The reason why these are interesting, is that we can split it in half!
- Let's grab just the bit in red, and the bit in green:

![[Pasted image 20231209125018.png]]

If we take an image and just put it through the first half (the [[Encoder]])... The thing that comes out the end of that green network is just 16,384 bytes. We have something that basically compresses an image, and the image that we get out the other end is some sort of compression.

How might this be useful?
- What about in an email situation? I have a big image, and I use my encoder half of the network to compress the image, and I send it to Roz. Roz uses the [[Decoder]] half of the AutoEncoder to "decompress" the image into (hopefully) the image that I originally meant to send! 

So we've created a compression algorithm! Pretty cool.
These compression algorithms work extremely well! Notice that we didn't train on just one image; we trained it on millions of millions of images!
If Roz and I both have a copy of this NN, we can share copies of images by sending just that compressed version of the image. Cool!

Where is this going?
- If the center of the autoencoder (the 16,384 bytes bit) contains all of the useful information of the image... Why on earth would we be training our UNet from before with the whole 786,432 bytes of information?
	- The answer is: We wouldn't! That would be stupid!

💡Instead we'll do our entire {train a model to generate handwritten digits from noise by learning how to identify noise, using a UNet} on the *encoded*, compressed bit of data from our AutoEncoder!

If we want to train our UNet on 10,000,000 pictures, we first put all 10,000,000 pictures through the (trained) AutoEncoder's encoder, and then feed those smaller pictures into the UNet training process to train our UNet!
- That UNet will no longer take a somewhat noisy image... instead it takes a somewhat noisy [[Latent]] -- the output will still be the noise.
- We can then subtract the noise from the somewhat noisy latents, and that will give us the "actual" latents.
- We can then take the output of the UNet... and pass into our AutoEncoder's decocder! Recall that the decoder takes a small, Latent-sized tensor, and outputs a large Image!

Okay

This is called a [[Variational Autoencoder]] (VAE).
- VAE's decoder
	- input: A small, Latent-shaped tensor
	- output: A large image

You only need the encoder half of the the VAE if you're *TRAINING* your UNet
- I think the UNET training looks similar.
	- Now, instead of having a clean picture of a 9 that we'll add some noise to and ask the UNet to be able to predict the noise...
	- We instead have a clean Latent, and we add some noise to it, and then we ask the UNEt to be able to be able to predict the noise!
	- In this way, the UNet "learns" what a clean latent looks like (by learning what non-clean-latent *noise* looks like), and should be able to generate latents from pure noise through iterative inference, similar to how we were able to generate handwritten digits from pure noise through iterative inference.
If you want to just do *INFERENCE* like we did today, you only need the decoder half of the VAE.
- Here, you'd got a trained UNet that can already take random noise and produce a Latent of some sort.
- Now, we just take that produced latent and "biggify"/"uncompress" it using the Decoder half of our autoencoder!

This whole bit of using the AutoEncoder and the latents is entirely optional!
- The initial bit that we described with handwritten digits and adding noise, etc. Works totally fine to train a UNet! But it saves us a lot of time and compute to run the UNet on the smaller, AutoEncoder encoded/compressed versions of images than on the images themselves.















