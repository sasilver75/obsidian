
![[Pasted image 20240524113220.png]]
What do we need RL for, given the above objective?
- The $y$ above: We're sampling from the model at multiple timesteps, the next word to generate. This sampling process is not differentiable; we're choosing one word from the distribution over the entire vocabulary over and over, until the end of sequence is finally sampled, which is then when we determine reward, etc. This is what stops us from doing our normal SFT finetune -- we're nto trying to maximize the likelihood of some gold output, we're trying to maximize the reward of some generation of our model. We don't know what part of the $y$ is good or bad; we only observe the reward at the END of all of the next-token-prediction steps ([[Credit Assignment Problem]]).

It can difficult to figure out what part of the output is good or bad.

----

But let's talk about an alternative, [[Direct Preference Optimization]] (DPO):
- No explicit reward model needed
- Not going to sample outputs y|x from the model
	- (In RL terminology, these generations are called "Rollouts")

DPO is one of an emerging class of methods called preference finetuning, a class of techniques for finetuning a model taking into account human preferences over generations.

![[Pasted image 20240524113955.png]]
They introduce a policy that is a function of the reference policy as well as the KL-modulated reward term -- this 1/B(r(x,y)).

Let's introduce a new policy $\pi^*$ that we'll later solve for as the optimal policy that minimizes the objective, and incorporates the reward term as well as $\pi_{ref}$ 

![[Pasted image 20240524114255.png]]
So what they do is take this minimization objective we previously had, and rewrite it to include this z(x) term, using some basic algebra
- Substitute z(x) into our objective. 
- After re-arranging terms, we get:

![[Pasted image 20240524114422.png]]
This is equivalent to the one above, but we've introduced z(x) into the equation.

But wait, if we actually look at the denominator term, we see that we can rewrite it as $\pi^*$ 
![[Pasted image 20240524114513.png]]
That term that's the  ratio of the two policies is (?) the KL divergence of the two distributions (our current policy pi, and our optimal policy pi-star).

![[Pasted image 20240524114745.png]]



