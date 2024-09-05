https://www.youtube.com/watch?v=Cu-aSZqxkZw&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5&index=9

In general, the topic of ML security is hot these days; let's touch on a bunch of topics from that area in this lecture, in addition to focusing on some data-centric attacks.

Imagine that we have a model behind a Web API:
If there's sensitive data input to the model training procedure, it's reasonable to think that the model might be influenced in some bad way by that sensitive data... and that perhaps it will leak it, or its presence will be knowable!


==[[Membership Inference Attack]]==
- Just given the model, it's possible to now, given a fresh data point, infer whether or not that datapoint was used to train the model.
- This is important for discovering copyrighted material, and worrisome for medical datasets.

==[[Data Extraction Attack]]==
- The idea is that, just given a trained ML model, you can extract raw training data from the model itself; the model sometimes memorize portions of training data, and we can pick that out from the model.

There are many other attacks, and thousands of research papers on the topic of ML security.

---


Defining Security of ML Systems
- ==[[Security Goal]]==: What should, and shouldn't happen with the system? What is the system trying to accomplish?
- ==[[Threat Model]]==: Constrains the adversary; what can/can't the adversary *do*? What does the adversary *know*? How much *computing power* does the adversary have?

==Only once you define these two things can you talk about whether or not the system is secure!==


Example: Image Prediction API (eg Google Cloud Vision AI)
- We feed in a picture of a person to some model that lives in the cloud, which outputs something like "Person 97%, Dog 2%, Cat 1%", and then you get back this set of predictions.
- Security Goals:
	- Model parameters kept secret from users
- Threat Model:
	- CAN
		- Adversary can issue queries to the model
			- Maybe the model accepts arbitrary JPEG files; they don't have to send natural images, they can construct whatever image they want, setting the color of each individual pixel.
		- The adversary can know the model architecture (eg because the model we're using has an academic paper published for it)
	- CAN'T
		- Adversary is NOT allowed to examine hidden layer activations
		- Any input to the model has to be value (eg input values between 0-255)
		- Let's assume the adversary makes <1000 queries per hour, because of rate-limiting
		- Let's assume the adversary spends less than $100k (limiting compute).


Example: Let's supposed we have  hospital that's trained a model on patient medical records, and it does something like output a classification result. Let's say that they've made this model available to the public (open weights).
- Security Goal:
	- Dataset is private and contains patient information; it shouldn't leak that information!
- Threat Modeling:
	- Since we're releasing the model online, the adversary has full access to the model -- this is called [[White-Box Access]] to the model, with full transparency.
	- Hospital claims that the adversary does'nt have access ot the original dataset on which the model was trained, but perhaps the adversary can access similar de-identified datasets released by other hospitals. 
		- So the adversary has access to the general data distribution from which our hospital's data was drawn.
	- The adversary can't in any way obtain some subset of the hospital's training data (eg by hacking into hospital servers).
	- The hospital probably trained it over a bunch of epochs, and took a bunch of checkpoints as it was training; we assume that the adversary does *not* have access to these intermediate checkpoints.



## Membership Inference Attacks
- Problem Setup for [[Membership Inference Attack]]s
	- In this attack, we have a dataset $\mathcal{D}$ containing  (x,y) datapoints, and we have a model $\mathcal{M}$ trained on this data.
	- Given some new datapoints ($x_*,y_*$)  and [[Black-Box Access]] to the model $\mathcal{M}$, what's the probability that the dataset $\mathcal{D}$ contains our $(x_*, y_*)$

One algorithm for doing this type of attack involves something called [[Shadow Model]]s
- Given an (x,y), given a model M, we feed x into the model to get $\hat{y}$, which is a vector of probabilities.
- We train an attack model A that takes as input both the output of the model and the original label ($\hat{y}$ and y) and tries to estimate the probability that our (x,y) are in D.

The question is: How do we actually construct this model A, using supervised learning? How do we get the training dataset for this model?

Step 1: Collect training data
- Even if we don't have access to the original training dataset $D$, we can construct a new dataset $D'$ that has a bunch of datapoints that kind of look like the stuff in $D$. In our case, we don't know what the hospital dataset looks like, but we can go on Kaggle and pull some de-identified patient information to construct $D'$.
- We partition this $D'$ into two dataset:
	- $D'in$
	- $D'out$
- We guess some model architecture, and train a shadow model $M_{shadow}$ on $D'in$.
	- We have access to the ground truth of what datapoints the model was trained on
- So now we can construct the training dataset for my model $A$ by, for each datapoint (x,y) in $D'in$, we can calculate our shadow model's prediction on x: $M_{shadow}(x) = \hat{y}$ .
	- We can then add to this new dataset we're constructing $D_A$ the datapoint $((\hat{y},y), yes)$. 
- We can do the same thing for every $x,y$ in $D_{out}$ : We give to our shadow model the x, to get a $\hat{y}$
	- And then add this to the $D_A$ $((\hat{y},y), no)$
- Now we have a new dataset $D_A$ that consists of prediction vector and class labels, and whether or not the thing was used in actually training the model.
- After that, the rest of the attack is pretty easy! 
- We train a model $A$ on our new dataset $D_A$ (shadow model should probably be an architecture that's similar to the one used by the victim, but the model A can pretty much be anything; simple models often work well)
- Perform the attack.

Downsides of this attack:
- Assumes that you can find some $D'$ dataset that's similar to the $D$ used to trained your target model.
- The original model might be really complicated, so it would cost a lot of compute to train your $M_{shadow}$ model.

As a result, there was followup series of methods:

[[Metrics-Based Attack]]
- Instead of training a shadow model ($), does simple things like feeding data points to model, getting output prediction vectors $\hat{y}$, and computing simple metrics based on that.
	- One example would be looking at *Confidence*; $\hat{y} = M(x)$, and then $score=max(\hat{y})$ -- how confident is the model in its output? This works pretty well! How do we choose this threshold, though? Some details are tricky to get right.


-----

## Data Extraction Attacks
- Neural networks often unintentionally memorized aspects of data!
- Given a model, can we figure out what points were used in the training dataset, without having a specific datapoint in mind? Given $M$ , find some $X \in D$ 

We can attack it with prompt engineering:
`"My social security number is: "` ... will the model output something sensitive?

Carlini et al, 2021 (Can't remember attack name):
1. Sample a bunch of data from the model
2. Do a membership inference attack (figure out which of the generations were in the training set)
![[Pasted image 20240701135236.png]]
In this attack, we use a pretty straightforward metrics-based approach. We use $p(x_i|x_1,...,x_i-1)$ , specifically [[Perplexity]] as a score to evaluate the generations (the authors experiment with multiple metrics).
$exp((-1/n)\sum_{i=1}^n{log(p(x_i|previous))})$ 

The authors used this to attack (in 2021) GPT-2, and found that they could extract hundreds of memorized training examples from GPT-2, and confirmed this with the authors at OpenAI.

