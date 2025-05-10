
![[Pasted image 20250118122038.png]]
Above: The above description is for a multi-armed bandit scenario, where the State only has one state, so we're just talking about action values.

Note that there's always at least one action with the optimal regret, which is going to be zero.

The more regret we have, the worse we're doing.

We might think about minimizing our total regret:
![[Pasted image 20250118122405.png]]
Typically our policy will depend on our history.
- We take an action and get a random reward from the action-value distribution.
- We use that reward to pick the next action, meaning the next action is somewhat random as well.

==We want to maximize cumulative reward, which is equivalent to minimizing total regret.==
- If we maximize the expected cumulative reward, we minimize the gaps between the optimal expected value and the one for the action we've selected...

This notion of regret is quite common in [[Multi-Armed Bandit]], where we reason about how quickly this regret grows. We'd like to find algorithms whose regret doesn't grow too quickly over the lifetime of learning!

![[Pasted image 20250118141633.png]]
We see that theoretically, for ==ANY ALGORITHM==, regret glows at least logarithmically in the number of timesteps. $logt$ depends on time; see the other stuff behind it doesn't depend on time, just on the difference between the optimal action and the action we're taking (regret) $\triangle_a$ and the [[Kullback-Leibler Divergence|KL-Divergence]] between the distribution between the reward under action $a$ and the distribution of reward under the action $a*$ .

Question: ==Can we find algorithms for which not just the lower-bound is logarithmic, but the upper-bound (the worst the algorithm will do) is ALSO logarithmic in expectation?==
- There are algorithms for which this is true!

![[Pasted image 20250118142112.png]]
For each action, let's consider how often we take the action, and how bad it was. We want this to be low -- we want the total regret to be small. A good algorithm ensures small counts when the regret is large, and for small-regret, the count of actions can be bigger! (I.E. we don't want to over-explore bad actions, or under-exploit good actions).

We want to implement: ==Optimism in the face of uncertainty== (which can be implemented in many ways)
- Whenever we're unsure about the value of an action, we want to be optimistic about it perhaps being good, and we want to pick it more often if we're more uncertain about its value.

![[Pasted image 20250118142254.png]]
Here are some belief probabilities that we have about the mean of some actions -- we're pretty sure that the expected value of the red action is positive, and not larger than two. It might be below zero.
For the blue action, we're a little more uncertain -- we think it's close to 0, but it might be over at -2 or 2 with smaller probability. The green action we've barely ever selected. We think it's expected value is somewhat more like -.25, but it might be as large as 4, since we've only selected it once or twice.
- So which action should we pick?
- ==If we have more uncertainty about the value, it's more important to explore that value!==
- As we explore these, the distributions are likely to get more and more narrow.

Maybe eventually we find that the mean value for the green distribution is actually larger than it is for the red distribution.
This motivates several algorithms, including the [[Upper Confidence Bound|UCB]] algorithm;
- The idea: Have an upper confidence bound $U_t(a)$ for each action value q(a), such that $q(a) \leq Q_t(a) + U_t(a)$ with high probability...
	- But we don't want to pick it to be too large so it becomes meaningless, but we want it large enough such that we're still certain that the mean is lower than this number, but small enough that this eventually converges to the true mean.
- Then, we select our actions greedily... but not respect only to Q_t(a), but to Q_t(a) + U_t(a)!
	- Essentially, we're going to ==define a time-and-action-dependent bound that's defined such that if we're very uncertain about an action, we'll pick it!== The only other reason why we might pick an action is because the estimate itself is high! The uncertainty should depend ont he number of times that $N_t(a)$ has been selected.
		- A small $N_t(a)$ should lead to a large $U_t(a)$ (because estimated value is uncertain)
		- A large $N_t(a)$ should lead to a small $U_t(a)$ (because we've seen many examples of this action being selected)
	- If the count becomes very large relative other actions, the bound should become small relative to other actions.
	- If we apply this algorithm, each action a only gets selected either:
		- $Q_t(a)$ is very large (it's proven so far to be a good action) or
		- $U_t(a)$ is very large (we're highly uncertain) or both!
	- So we'll be selecting actions that we think are good or actions that we think are underexplored. ==If we know that an action ahs a comparatively low value, we should explore it less; this is where [[Epsilon-Greedy]] fails! We keep exploring actions even if we've explored them often and are quite certain about the value!==
![[Pasted image 20250118144225.png]]
So how can we find this bound $U_t(a)$?

We're going to use something called a concentration inequality; we're going to use Hoeffding's Inequality, which gives us a bound on how wrong our estimate is:
![[Pasted image 20250118144259.png]]
Think of X_1 ... X_n as being the rewards from a specific action, drawn randomly from the same distribution. Let's assume that they're all between 0 and 1.
- There's some true mean $\mu$ for the expected value of random variables in this set.
- There's a sample mean $\bar{X}$ for the samples that we've seen so far.
- Hoeffding's Inequality tells us bonds for how off our sample mean could be from the true mean.
- Consider a quantity u:
	- If we add it to the mean we estimate so far
	- How likely is it that Xbar + u is still less than the true mean? 
	- We can bound this as the right side of the inequality that we see above.
- This is typically going to be a small number, and it's smaller the larger n is or u is
	- The more numbers there are in the average, the less likely we are that if we add an amount, it's still smaller than the actual mean.  We're quite likely to be within u of the actual mean, if we have enough numbers in our sample. Similarly, if we picked our u to be larger, then the probability also decreases.

The idea is to apply this to bandits with bounded rewards.

![[Pasted image 20250118145915.png]]
So we havae a bound on a probability; let's pick $u$ so that it doesn't exceed some number?
Now we can pick a probability $p$, and pick our bound to be equal to that probability $p$
- We can then solve this for the boundary $U_t(a)$.

Then we know the probability that we have a sample mean that's further away from the population mean than this bound $U_t(a)$ is $p$.

And now we can pick $p$ to be small and decrease over time!
![[Pasted image 20250118150041.png]]
We reduce the probability that we'll be more than this bound off.
- We can pick p to be 1/t, plug it into the bound and get one that looks like this ⬆️.

We didn't say how to pick p
We didn't say why it should be p=1/t
But the idea is that we can pick the probability that we're goign to be more than this bound U wrong with our sample estimates.
- If n is pretty small, this bound will be pretty large
- If n is large, this bound will be pretty small

- Indefinitely, we continue to select an action, but less and less so.
- ==This ensures that we keep exploring, but not too much!==

So the [[Upper Confidence Bound|UCB]] algorithm looks like this:
![[Pasted image 20250118150456.png]]
We've introduced a new hyperparameter c; within the square root in the last slide we were dividing by 2; that's the same as c -- it's just a hyperparameter; larger c's explore more, smaller c's explore less; c=0 gives you the greedy algorithm.
- If your gap/regret of an action is large, then your N_t(a) is small, because your Q_t(a) is likely to be small.
	- Meaning for high-regret actions with low estimated reward, based on experience, we will likely not explore them often. 
- ==(Re last bulletpoint on image): We have established an algorithm which has logarithmic regret as an upper bound! We also know that the lower bound is logarithmic, so we know that our algorithm's regret is going to be logarithmic in time.==

Instead of taking UCB and proving it, let's try to derive it from first principles!

-----

From : https://youtu.be/aQJP3Z2Ho8U?si=Jkixv2O3Tn5fQyNz

We'll discuss several algorithms:
- Greedy
- [[Epsilon-Greedy]]
- [[Upper Confidence Bound]] (UCB) algorithm
- [[Thompson Sampling]]
- [[Policy Gradient]] methods

The first all use action-value estimates $Q_t(a) \approx q(a)$ 
- q(a) is the true expected reward, and Q_t is an approximation of this

![[Pasted image 20250118122749.png]]
We're picking out with an indicator function the timestamps with which we selected action $a$. This is just taking the average of returns in states where we took action $a$.

We could also do this incrementally:
![[Pasted image 20250118122847.png]]
We take our action value at timestep t and define it to be the action value at a previous timestep plus some learning rate $\alpha$ times an error term (the reward we received given that we took the action minus the estimated value of that action so far); We move our estimate a little towards reality.
- We *could* define our step size to be inversely related the number of times we've taken a certain action.
	- It could be a constant alpha, which would lead to "Tracking" behavior, rather than averaging -- this is useful for nonstationary targets.

### Greedy
- Takes the action with the highest value as we understand it so far:
	- $A_t = \underset{a}{argmax}Q_t(a)$
- Given two actions A and B (A is secretly better), the policy might randomly first choose A, getting some reward. Now the expected action value of action A is higher than action B (which is 0), so the greedy policy just keeps selecting A forever.
- This leads to linear expected total regret, with the ==policy getting stuck on a suboptimal action forever!==
- The greedy policy doesn't explore enough; it keeps exploiting the knowledge that it has, and doesn't consider exploration.
- The alternative is to add some noise to keep it exploring!

### Epsilon-Greedy
- With probability 1- epsilon we select the greedy action
- With probability epsilon we select  a random action
![[Pasted image 20250118124323.png]]
- [[Epsilon-Greedy]] greedy continues to explore with this epsilon, but it also has linear expected total regret, because it the epsilon doesn't decrease by default, because you keep picking exploratory actions even after the point where you've obtained enough information about bad actions. 


### Policy Search
- Can we learn policies $\pi(a)$ directly instead of learning values?
- For instance, we could define ==action preferences== $H_t(A)$ and a policy that's differentiable so we can take a gradient; we use a softmax (we exponentiate the preferences and then just normalize to get our probability distribution over actions)
	- The exponentiating gets us a positive number
	- Because we're normalizing, the probabilities must sum to one
- We select an action according to this policfy! Note that the preferences themselves are NOT supposed to be action values (q(a)) per-se, though they could be; we consider them to just be learnable policy parameters.
![[Pasted image 20250118124524.png]]
One algorithm to do this is called [[Policy Gradient]] algorithms:
- Idea: We update our policy parameters (our learnable prefernces) to be something that we can learn using gradient ascent!

Our parameters theta are a vector containing our action preferences for each of our actions... this is quite easy to apply at scale using deep learning, where theta are the parameters of our DNN.
We for now can consider theta to be a vector of our action preferences for now, though.
We want to do gradient ascent on the *value of our policy* (the expected return given that we're following the actions prescribed by the policy).
![[Pasted image 20250118124717.png]]
Problem: We have a gradient of an expectation... but we don't know that expectation, usually!
- We'll use a trick that lets us get the stochastic sample for this.
- The sample for this expectation would just be a reward, and we don't know how to take the gradient of the reward with respect to the policy parameters directly anyways.
- So we need to do a little bit of work to make this isn't something that we can sample from so that we can do stochastic gradient descent.

We can turn that gradient into something that we can sample, this is sometimes called the [[Log-Likelihood Trick]], or the REINFORCE Trick. 

![[Pasted image 20250118124909.png]]

We start with the gradient of the expectation of the reward, given our policy, which is the thing that we'd like to be able to solve for, but we don't know the expected return :\
- We expand this into the summation over the actions given the policy times the expected reward given that we've taken specifically that action. We know this to be our action-value q(a).
- We know that our q(a) doesn't depend on our policy parameters, because we've already pinned down our action by now.
- This means that we can push our gradients inside the summation, and it will only affect our policfy, not hte q values.
- We're going to multiply this whole quantity by something that's basically one, the probability of the policy selecting an action divided by the probability of the policy selecting an action.
	- We push the division to the back of the notation, and then we see that we have a form that we can write as an expectation again!
- This can be written s the expectation of the reward ... multiplied by a weird-looking ratio, which is the gradient of the probability of selecting action A_t divided by that same probability.
- We can then write this whole term equivalently as the expectation of the reward times the gradient of the *logarithm* of the policy. This is true because of the chain rule:
	- The gradient of the logarithm of {something} is 1/{something}; but then we have to apply the chain rule... Because we have the gradient of the log of some f(x)/x, so the gradient is 1/x * gradient of the function of x. (??)

==The important thing is that we've arrived on the right side to something that has an expectation on the outside and then some term on the inside which has the gradient... This means we now have something that we can sample!==
- We can get rid of the expectation in our updates by just sampling the thing inside and following the ==stochastic== gradient instead of the true gradient.
- We know that stochastic gradient descent/ascent still works quite well!


![[Pasted image 20250118125611.png]]
We can turn this into a stochastic gradient ascent algorithm, where we just take samples of that thing in the middle!
- Because we want to maximize this expected reward by changing our policy parameters theta, we do gradient ascent. We use a learning rate because we're doing updates using stochastic estimates of the true gradient, and over time we wander towards something good!

We execute this over and over again, performing stochastic gradient ascent to optimize the expected return!
- We can use sampled rewards, we don't need value estimates!

Let's extend it to bandits:
### Gradient Bandits

If we parameterized our policy using those action *preferences* H that we talked about earlier...
![[Pasted image 20250118131314.png]]
- We only have a single parameter in theta for a given action a.
- We add some step size alpha times the return that we've seen, times the gradient of the selected action A_t.. this turns out to be 1 minus the probability of he action that we selected
- Rewritten, this means that the preference for action A_t that we selected gets updated by adding learning rate times reward times (1-probability of selecting action), while the preferences of all the OTHER actions get updated by subtracting a learning rate times reward times probability of selecting that action.

Now that we know these updates, can we interpret them somehow?
- Let's say we saw a reward of +1:
	- The preference for the action that we selected gets updated to be higher, because the step size is a positive quantity
		- Alpha is positive
		- R_t is positive (1)
		- $1-  \pi_t(A_t)$ for the selected action was clearly positive
	- So if the reward is positive, the preference for the selected action will increase.
	- And the preference for all selected actions will go down
		- Note also that how much they go down will depend on the probability that they were going to be selected, $\pi_t(a)$. 
		- Unselected actions that were more likely to have been selected take a larger "hit" than actions that were less likely to be selected, since the delta is $\alpha R_t \pi_t(a)$.
- So what happens if we had taken an action and gotten a negative reward? The opposite would have happened! The preference for the action that we took that got a negative reward will go down, and the preference for the other actions will go up.
- But what if there are only positive rewards? This will still work!

Intuition: ==The preferences for actions with higher rewards increase more (or decrease less, net), making them more likely to be selected again==.
- The exploration though isn't very explicit; it's just due to the policy being stochastic.
- Because it's a gradient algorithm, it can get stuck in a local optimum; there's no guarantee that it won't suffer from linearly-increasing regret over time, since it can get stuck in a suboptimal policy.
- Still, we can extend this algorithm quite easily to DNNs.

... More continued


![[Pasted image 20250118141633.png]]
We see that theoretically, for ==ANY ALGORITHM==, regret glows at least logarithmically in the number of timesteps. $logt$ depends on time; see the other stuff behind it doesn't depend on time, just on the difference between the optimal action and the action we're taking (regret) $\triangle_a$ and the [[Kullback-Leibler Divergence|KL-Divergence]] between the distribution between the reward under action $a$ and the distribution of reward under the action $a*$ .

Question: ==Can we find algorithms for which not just the lower-bound is logarithmic, but the upper-bound (the worst the algorithm will do) is ALSO logarithmic in expectation?==
- There are algorithms for which this is true!

![[Pasted image 20250118142112.png]]
For each action, let's consider how often we take the action, and how bad it was. We want this to be low -- we want the total regret to be small. A good algorithm ensures small counts when the regret is large, and for small-regret, the count of actions can be bigger! (I.E. we don't want to over-explore bad actions, or under-exploit good actions).

We want to implement: ==Optimism in the face of uncertainty== (which can be implemented in many ways)
- Whenever we're unsure about the value of an action, we want to be optimistic about it perhaps being good, and we want to pick it more often if we're more uncertain about its value.

![[Pasted image 20250118142254.png]]
Here are some belief probabilities that we have about the mean of some actions -- we're pretty sure that the expected value of the red action is positive, and not larger than two. It might be below zero.
For the blue action, we're a little more uncertain -- we think it's close to 0, but it might be over at -2 or 2 with smaller probability. The green action we've barely ever selected. We think it's expected value is somewhat more like -.25, but it might be as large as 4, since we've only selected it once or twice.
- So which action should we pick?
- ==If we have more uncertainty about the value, it's more important to explore that value!==
- As we explore these, the distributions are likely to get more and more narrow.

Maybe eventually we find that the mean value for the green distribution is actually larger than it is for the red distribution.
This motivates several algorithms, including the [[Upper Confidence Bound|UCB]] algorithm;
- The idea: Have an upper confidence bound $U_t(a)$ for each action value q(a), such that $q(a) \leq Q_t(a) + U_t(a)$ with high probability...
	- But we don't want to pick it to be too large so it becomes meaningless, but we want it large enough such that we're still certain that the mean is lower than this number, but small enough that this eventually converges to the true mean.
- Then, we select our actions greedily... but not respect only to Q_t(a), but to Q_t(a) + U_t(a)!
	- Essentially, we're going to ==define a time-and-action-dependent bound that's defined such that if we're very uncertain about an action, we'll pick it!== The only other reason why we might pick an action is because the estimate itself is high! The uncertainty should depend ont he number of times that $N_t(a)$ has been selected.
		- A small $N_t(a)$ should lead to a large $U_t(a)$ (because estimated value is uncertain)
		- A large $N_t(a)$ should lead to a small $U_t(a)$ (because we've seen many examples of this action being selected)
	- If the count becomes very large relative other actions, the bound should become small relative to other actions.
	- If we apply this algorithm, each action a only gets selected either:
		- $Q_t(a)$ is very large (it's proven so far to be a good action) or
		- $U_t(a)$ is very large (we're highly uncertain) or both!
	- So we'll be selecting actions that we think are good or actions that we think are underexplored. ==If we know that an action ahs a comparatively low value, we should explore it less; this is where [[Epsilon-Greedy]] fails! We keep exploring actions even if we've explored them often and are quite certain about the value!==
![[Pasted image 20250118144225.png]]
So how can we find this bound $U_t(a)$?

We're going to use something called a concentration inequality; we're going to use Hoeffding's Inequality, which gives us a bound on how wrong our estimate is:
![[Pasted image 20250118144259.png]]
Think of X_1 ... X_n as being the rewards from a specific action, drawn randomly from the same distribution. Let's assume that they're all between 0 and 1.
- There's some true mean $\mu$ for the expected value of random variables in this set.
- There's a sample mean $\bar{X}$ for the samples that we've seen so far.
- Hoeffding's Inequality tells us bonds for how off our sample mean could be from the true mean.
- Consider a quantity u:
	- If we add it to the mean we estimate so far
	- How likely is it that Xbar + u is still less than the true mean? 
	- We can bound this as the right side of the inequality that we see above.
- This is typically going to be a small number, and it's smaller the larger n is or u is
	- The more numbers there are in the average, the less likely we are that if we add an amount, it's still smaller than the actual mean.  We're quite likely to be within u of the actual mean, if we have enough numbers in our sample. Similarly, if we picked our u to be larger, then the probability also decreases.

The idea is to apply this to bandits with bounded rewards.

![[Pasted image 20250118145915.png]]
So we havae a bound on a probability; let's pick $u$ so that it doesn't exceed some number?
Now we can pick a probability $p$, and pick our bound to be equal to that probability $p$
- We can then solve this for the boundary $U_t(a)$.

Then we know the probability that we have a sample mean that's further away from the population mean than this bound $U_t(a)$ is $p$.

And now we can pick $p$ to be small and decrease over time!
![[Pasted image 20250118150041.png]]
We reduce the probability that we'll be more than this bound off.
- We can pick p to be 1/t, plug it into the bound and get one that looks like this ⬆️.

We didn't say how to pick p
We didn't say why it should be p=1/t
But the idea is that we can pick the probability that we're goign to be more than this bound U wrong with our sample estimates.
- If n is pretty small, this bound will be pretty large
- If n is large, this bound will be pretty small

- Indefinitely, we continue to select an action, but less and less so.
- ==This ensures that we keep exploring, but not too much!==

So the [[Upper Confidence Bound|UCB]] algorithm looks like this:
![[Pasted image 20250118150456.png]]
We've introduced a new hyperparameter c; within the square root in the last slide we were dividing by 2; that's the same as c -- it's just a hyperparameter; larger c's explore more, smaller c's explore less; c=0 gives you the greedy algorithm.
- If your gap/regret of an action is large, then your N_t(a) is small, because your Q_t(a) is likely to be small.
	- Meaning for high-regret actions with low estimated reward, based on experience, we will likely not explore them often. 
- ==(Re last bulletpoint on image): We have established an algorithm which has logarithmic regret as an upper bound! We also know that the lower bound is logarithmic, so we know that our algorithm's regret is going to be logarithmic in time.==

Instead of taking UCB and proving it, let's try to derive it from first principles!


### Bayesian Bandits

- In a Bayesian approach, we could keep track of a distribution over the expected value for each action.
	- We aren't trying to model the distribution of the rewards, instead we're gong to quantify our uncertainty about where the *expected reward* is, and update that over time.
		- The probability isn't on the reward, it's over the *expected reward*, which under the bayesian approach we consider to be a random quantity because we have uncertainty about what its true value is.
![[Pasted image 20250118151046.png]]
- We can use these beliefs to guide our exploration; we could very easily then pick the upper confidence intervals (?).

Example:
![[Pasted image 20250118151535.png]]
![[Pasted image 20250118151541.png]]
![[Pasted image 20250118151613.png]]


### Thompson Sampling
- [[Thompson Sampling]] is also an algorithm to solve bandits, and it's a [[Bayesian]] approach, in particular it's related to something that we'll describe first, called ==Probability Matching==.
- Probability Matching is quite different from [[Upper Confidence Bound|UCB]], because it's a random algorithm (our action will be a random action; our policy will be stochastic) -- w pick an action a according to the probability (belief) that a is optimal.
- ![[Pasted image 20250118152001.png]]
- This is somewhat unintuitive, but it's optimistic in the face of uncertainty (which we want), because if we have large uncertainty about where our action will be, then the probability of it  being maximal also goes up!
	- Maybe it's not the MOST likely action to be maximal, but it might have a *fair* chance of being maximal if your uncertainty is high.
- So actions have higher probabilities when:
	- The action is high-valued (estimated as)
	- When we have high uncertainty about them.
- This is similar to approaches like [[Upper Confidence Bound|UCB]]
- But it's not immediately obvious that this is the right probability that we should assign to an action... that picking based on the probability that you're the optimal action should be the probability that we should use for exploration 🤔.

There's an easier approach than solving $\pi(a)$ numerically; using [[Thompson Sampling]] (1933)!
![[Pasted image 20250118152155.png]]
- The idea is simple!
	- We keep track of posterior distributions, updating these via Bayes rule, for instance
	- We sample from each belief distribution an actual action value.
		- We have a belief distribution at timestep t about what we think the mean value for an action is...
		- We then sample that distribution, giving us an action value
		- We do that for each of our actions
		- Then we pick the greedy action according to the sample action values.
- It turns out that if you do this, Thompson Sampling will sample identically to how Probability Matching would do!

Thompson sampling is a technique that lets us go from these bayesian probability distributions to a policy.
Interestingly, Thompson sampling achieves Lai and Robbins lower bound on regret (logarithmic), and is therefore optimal, just like UCB.
- ==Thompson Sampling is considered, similar to UCB, one of the optimal algorithms.==
- Not all of these approaches are easy to scale to the full RL case, which is why we're covering multiple of these algorithms.


---------

## Planning to Explore
- We've been viewing bandits as ==one-step== decision-making problems, but we can also view them as ==sequential== decision-making problems, but instead of having the environment state be sequential, we'll talk about the internal state of the agent.
- At each timestep, the agent updates some internal state to summarize the past (this doesn't need to includei nformation about the environment, but should contain information about the actions and rewards)
- So taking actions cause transitions to new ==information states== S_t+1 with some probability
	- It's random because it depends on the reward that we receive.

![[Pasted image 20250118154040.png]]
So even in Bandits, we can think of actions as effecting the future after all, but not because they effect the state of the environment, but because they affect the future because of how they effect the internal state of the agent.
![[Pasted image 20250118154439.png]]








 
