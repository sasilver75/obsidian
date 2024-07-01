Proposed in 2016, and was one of the first times people thought about membership-inference attacks in neural networks.

Problem Setup for [[Membership Inference Attack]]s
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