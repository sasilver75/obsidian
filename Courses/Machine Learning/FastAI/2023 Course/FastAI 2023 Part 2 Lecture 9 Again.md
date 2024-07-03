
Lecture Date: October 2022

What it's doing in these 51 steps (that we'll cover soon) is starting with random noise, and each step trying to make it slightly ness noisy, and slightly more like the thing we want.
![[Pasted image 20240329181151.png|300]]

If you look closely, you can kind of see in the noise, something that looks vaguely like a 1 on the right side... and that's how these models basically work...

So you might ask: "Why can't we just do this in one step?"
- And the response is going to be: "If you try to do it in one go, it doesn't do a very good job! These models can't (in Oct 2022) be smart enough to do it in one go! But we _were_ successful in turning the 51 steps into (yesterday) 3-4 steps!"

Let's look at a few things that we can tune

Stable Diffusion built a `Diffusors` library, which we're using in our thing.

Let's create a grid of images
What we show: You can take your prompt and create four copies of the prompt... and then pass to the pipeline the prompts, and use a `guidance_scale` param which says: "To what degree should we be focusing on the specific caption, versus just creating an image." Generally 7.5 (At the time of lecture) is the default.
![[Pasted image 20240329181516.png|300]]

As we observe the results, we can see that there's an increasing adherence to the prompt as we increase guidance_scale.

[[Classifier-Free Guidance]] is a method to increase the adherence of the conditioning signal we used (the text).
- The larger the guidance, the more the model tries to represent the text prompt, and generally large values tend to produce less diversity.


There's something similar you can do, where you ask the model to generate two images, and subtract one from the other!
![[Pasted image 20240329183453.png|300]]
Give labrador, subtract the "caption: blue!"

This is called a [[Negative Prompt]]

So it takes the previous prompt and 
![[Pasted image 20240329183545.png|300]]


Something else you can play with...
- You don't have to just pass in text, you can also pass in images! For this you need a different pipeline (StableDiffusionImg2ImgPipeline), where you can grab a skethy-looking MS Paint sketch of a horse, and pass it into your  model

![[Pasted image 20240329183759.png|450]]
You can see the composition of the resulting image is the same!

So you can sort of use the image to guide the output (in terms of composition) of the image!
What you can do now... is take these output images, say "this middle one is nice," let's make *this* the init_image in our img2img pipeline, and do another prompt, saying "oil painting of wolf howing at moon," with this image!

![[Pasted image 20240329183914.png|450]]


You can also fine-tune a model to produce (eg) better pokemon images!
![[Pasted image 20240329184844.png|500]]

Trained on thousands of images of pokemon -- used a model called BLIP to generate captions for these images, and then fine-tuned a stable diffusion model using image/caption pairs, and took that fine-tuned model and passed it things like "girl with a pearl earring," "obama creature," "donald trump," etc. and got back these super nifty images that now reflect the fine-tuning dataset he used!! :) 


Fine-tuning took a lot of data and time, but you can do many types of fine-tuning!

[[Textual Inversion]] involves fine-tuning just a *single embedding!*  We can quickly "teach" a new word to the text model and plan its embeddings close to some visual representation -- this is achieved by adding a new token to the vocabulary, freezing the weight of the model (besides the encoder), and train with a few representative images.

We can create a new embedding... where we try to  make images that look like this :

![[Pasted image 20240329190132.png|300]]


We can give this concept a name -- call it "\<watercolor-portrait\>" And then we can basically add that token to the text model, and train the embeddings for this so that they match the example images that we've seen.
This is going to be much faster, because we're training just a single token, for four pictures!
Then later, we can say for our prompts:

"Woman reading in the style of \<watercoloor-portraits\>", and we'll get back a lovely new image!


![[Pasted image 20240329190255.png]]


[[Dreambooth]] is a kind of finetuning that attempts to introduce new subjects by providing just a few images of the new subject!
The goal is similar to Textual inversion, but different
- Take an existing token that isn't used much ("sks") and fineutnes the model to bring it close to the images we provide.
- So pedro took some images, and said: "Painting of sks person in the style of Paul Signac", after fine-tuning it with images of Jeremey Howard, under this "sks" token

![[Pasted image 20240329190602.png|400]]
Above:: Training with Dreambooth is finicky and sensitive to hyperparameters, since we're asking a model to overfit the prompt to the supplied images. In some situations, we observe problems such as catastrophic forgetting of the associated term ("person", in this case).
- The authors use a technique called "Prior Preservation" to avoid ti, by performing a special regularization using some other examples of the term, in addition to the ones we provided. The cool thing about this idea is that those examples can be easily generated by a pre-trained SD model itself!


### Stable Diffusion Explanation
- Stable Diffusion is usually focused on a specific mathematical derivation. We think of it in a different way! It's mathematically equivalent to the approach that you'll see in other places, but you'll realize and discover that it's conceptually much simpler!


(( Watching his SD explanation @ 45:00 ))
- Explanation of Fuzzing
- Explanation of SD, generally
- Explanation of uNet


### So what is Stable Diffusion?
- There are three main components!
	- ...