#article 
Link: https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of

----

![[Pasted image 20240209171224.png]]
Although it's useful for a variety of applications, [[Reinforcement Learning]] is a key component of the alignment process for LLMs due to its use in [[Reinforcement Learning from Human Feedback|RLHF]] -- but RL is less widely understood within the LLM AI community -- many practitioners are more familiar with supervised learning techniques, which creates a bias against using reinforcement learning despite its massive utility.

We'll be talking here about [[Proximal Policy Optimization]], or [[Proximal Policy Optimization|PPO]], which is heavily used in RLHF.

![[Pasted image 20240209171505.png]]
(Oct 2023 picture; Now, it probably contains DPO in the policy optimization side?)

There are two types of model-free RL algorithms: [[Q-Learning]]]] and Policy Optimization.

Let's now overview policy optimization and [[Policy Gradient]]s, two ideas heavily utilized by modern RL algorithms.

Notably, we'll look at [[Proximal Policy Optimization|PPO]], the most commonly-used RL algorithm for fine-tuning LLMs (until late 2023).

## Reinforcement Learning Basics

> RL is the study of agents and how they learn by trail and error. It formalizes the idea that rewarding or punishing an agent for its behavior makes it more likely to repeat or forego that behavior in the future.

In a prior overview in [[Basics of Reinforcement Learning for LLMs (Sep 2023) {Simon Wolfe, Deep Learning Focus Newsletter}]], we learned about the problem structure that's typically used for RL, and how this structure can be generalized to the setting of fine-tuning an LM.

Let's briefly overview the key ideas and introduce some new concepts related to policy optimization.

#### MDPs and Fundamental Components in RL
- The RL framework can be formalized as a [[Markov Decision Process]] (MDP), which has states, actions, rewards, transitions, and a policy.
![[Pasted image 20240209172027.png]]
For the purposes of this post, we'll assume that our ==policy== is a machine learning model with parameters ==θ==. 
- ==The policy takes a state as input and predicts some distribution over the action space==.
![[Pasted image 20240209203046.png]]
- By using our policy to predict each next action, we can traverse an environment, receive rewards, and form a sequential *trajectory* of states and actions.
- Typically, we refer to the entity traversing the environment as an *agent*, which implements the policy shown above when choosing each action.

![[Pasted image 20240209203600.png]]
Above: This is the process of our agent traversing the environment, formalized as a [[Markov Decision Process]].
- ((==Sam==)) Given a state, the agent uses the policy to generate a probability distribution over actions that it can take. It selects an option from that probability distribution. A transition function determines what state you land in after action `a` from state `s`. This state `s+1` may yield a reward from the environment. We accumulate all state/action pairs into a trajectory.

Reward and return
- As our agent traverses the environment, it receives positive or negative reward signals for the actions that it chooses and the states that it visits.
- Our goal is to learn a *policy* from theses reward signals that maximizes total reward across an entire trajectory sampled from the policy.
	- This idea is captured by the return, which sums the total rewards over an agent's trajectory:

![[Pasted image 20240209204648.png]]

Here, the return is being formulated with a ==discount factor==, but this is not always present or necessary.
- ==The two major types of returns considered within RL are the infinite-horizon discounted reward and the finite-horizon undiscounted reward.==

![[Pasted image 20240209204938.png]]

Value and Advantage Functions
![[Pasted image 20240209205037.png]]

One final concept that will be especially relevant to this point is that of a value function. ==In RL, there are four basic value functions (shown above)==, all of which assume an infinite-horizon discounted return. 

1. ==*On-Policy* Value Function==
	- Expected return if you start in state `s` and act according to policy $\pi$ afterwards.
2. ==*On-Policy* Action-Value Function==
	- Expected return if you start in state `s` , take some action `a` (which may or may not come from the current policy), and act according to policy $\pi$ afterwards.
3. ==*Optimal* Value Function==
	- Expected return if you start in state `s` and *always* act according to the *optimal policy* afterwards.
4. ==*Optimal* Action-Value Function==
	- Expected return if you start in state `s`, take some action `a` (which may or may not come from the current policy), and act according to the *optimal policy* afterwards.

There's an important connection between the optimal policy in an environment and the optimal action-value function -- namely, the optimal policy selects the action `a` in state `s` that maximizes the value of the action-value function.

![[Pasted image 20240209210832.png]]

Advantage functions
- Using the value functions described above, we can define a special type of function called an ==advantage function==, which is heavily used in RL algorithms that are based on policy gradients.
- Simply put, the *advantage function characterizes how much better it is to take a certain action `a` relative to a RANDOMLY-SELECTED action in state `s`, given policy $\pi$ .

> The value of your starting point is the reward you expect to get from being there, plus hte value of wherever you land next.

Connection to the [[Bellman Equation]]:
- Finally, we should note that each of the value functions have *their own Bellman equation* that quantifies the value of a particular state or state-action pair in RL.
- Bellman equations are the foundation of RL algorithms such as [[Q-Learning]] and [[Deep Q-Learning]].

### Policy Optimization
- Now let's explore the basic idea behind policy optimization -- and how this idea can be used to derive a policy gradient (and several variants of this).
- Our goal will be to find parameters $\theta$ for our policy that maximize the objective function below:
![[Pasted image 20240209212217.png]]
- Learn a policy that, when we use it to create a trajectory, maximizes the reward of that trajectory.
	- In words, this objective function measures the expected return of trajectories sampled from our policy within the specified environment.

If we want to find parameters $\theta$ that maximize this objective function, one of the most fundamental techniques that we can use is *gradient ascent*, which iterates over parameters $\theta$ using the update rule shown below.

![[Pasted image 20240209212907.png]]
Above: ((Does this make sense? How do you take the gradient of an objective function with respect to the parameters -- isn't the reward system not differentiable, or something, if it's just human feedback or gridworld feedback? Hmm...))

Gradient ascent/descent is a fundamental optimization algorithm that (along with its many variants) is heavily used for training ML models across a variety of circumstances.

Gradient Ascent/Descent algorithm:
1. Compute the gradient of objective with respect to current parameters.
2. Multiply this gradient by the learning rate.
3. Tweak the parameters by the addition/subtraction of this scaled gradient.

After successive applications of this gradient ascent update, we should move our policy towards the optimal policy by increasing the desired objective. ==This is the fundamental idea behind Policy Optimization==.
- (Note: the number of steps that you might need to take to get there varies by your objective and environment. Some landscapes don't have a single local minima, and are harder to optimize to the global maximum.)

The number of required updates to reach convergence

### Deriving and using a basic policy gradient
![[Pasted image 20240209213849.png]]
- But how do we actually compute the gradient of our objective? To answer this, we need a bit of math. We'll outline the basic ideas for how to do this here, but won't go too far in depth.

...

![[Pasted image 20240209214510.png]]

![[Pasted image 20240209214319.png]]
- Because our policy can have a potentially infinite number of trajectories, we have to express this operation as an integral instead of a discrete sum over trajectories.

From here, we can notice that our expression depends on two quantities:
1. The return of a trajectory
	- We just get this from our environment for a trajectory.
2. The probability of that trajectory under our current policy
	- We can compute the probability of a trajectory under the current policy as shown below.

![[Pasted image 20240209215122.png]]
Above: The probability of a trajectory of T steps is the probability of starting at state `s[0]` and multiplying it by the probability of landing in each state along the way, at every time step, given your policy.
- In reality, you were in states s0, s1, s2, ..., st that corresponded with takings actions a0, a1, ... at at the corresponding states. What's the probability of having that state/action trajectory?
- At every step, you sampled your policy and got a distribution of actions. That policy assigned some probability to each action. How much did it assign to the action that we ultimately took? And given that we took that action, what's the probability that taking action `a` at state `s` lands us specifically in the state `s[t+1]`? We do this same check for every "layer" of the tree of possibilities.

Above, we're using the ==chain rule of probability== to derive the probability of the overall trajectory under the current policy.
Then, by combining the expression we've derived so far and applying the ==log-derivative trick== ([link](https://andrewcharlesjones.github.io/journal/log-derivative.html), we arrive at the expression shown below:

![[Pasted image 20240209220559.png]]

- Now, we have an actual expression for the gradient of our objective function that we can use in gradient ascent!
	- Plus this expression only depends on the return of a trajectory and the gradient of the log probability of an action, given our current state.
	- As long as we instantiate our policy such that the gradient of action probabilities is computable (which is easy to do if our policy is a neural network), we can easily derive both of these quantities.

#### Computing the policy gradient in practice
- Computing the expectation used in the expression above analytically would require an integral, but in practice we can just estimate the value of this expectation by sampling a fixed number of trajectories.
- In other words, we can just:
	1. Sample several trajectories by letting the agent interact with the environment according to the current policy.
	2. Estimate the policy gradient using an average of relevant quantities over the fixed number of sample trajectories.

![[Pasted image 20240209220919.png]]

We compute this policy gradient shown above in every training iteration, or gradient ascent step.

### An implementation
- Now that we have a basic understanding of policy gradients, we can look at an example implementation... A great example of policy gradients is provided in OpenAI's spinning up tutorial series for RL.

#### Variants of the Basic Policy Gradient
- There are several variants of the policy gradient that can be derived; each of them addresses issues associated with the simple policy gradient we learned in th previous section.

![[Pasted image 20240209221425.png]]
==Reward-to-go trick==
- Our initial policy gradient expression (above) increases the probability of a given action based on the total return of a trajectory, which is a sum of (possibly discounted) rewards obtained over the *entire trajectory*.
- Consider: *Why should we consider rewards that are obtained BEFORE this action is ever taken? Shouldn't we only encourage actions based on rewards obtained after they're taken?*
	- Answer: Yes! This simple change leads to a new variant of the policy gradient expression referred to as the "reward-to-go" policy gradient.
![[Pasted image 20240209221716.png]]
Above: We can derive the reward-to-go policy gradient using the expected grad-log-prob (EGLP) lemma. 

We can go further with the EGLP lemma and use it to show that the modified expression above also maintains the desired expectation of the policy gradient while reducing the variance
![[Pasted image 20240209221924.png]]
We add a baseline function to our expression that only depends on the current state. There are several useful functions that we can consider as a baseline.


#### Vanilla Policy Gradient
![[Pasted image 20240209222010.png]]

The ==vanilla policy gradient== has a similar structure to the formulations above but uses the advantage function as shown below (again, this maintains the same expectation while reducing variance, meaning we can accurately estimate the policy gradient with fewer sample trajectories.)

![[Pasted image 20240209222129.png]]
Similar to the other policy gradient algorithms, we can estimate the above expression using a sample mean and optimize a policy with this gradient via gradient ascent.
- Vanilla policy gradient is an on-policy algorithm so we just do this by allowing the current policy to interact with the environment and collect a sufficient amount of data.


Connection to language models:
- Formulating the policy gradient with an advantage function is extremely common.
- Some RL algorithms that are commonly used for finetuning models (like trust region policy optimization (TRPO) and [[Proximal Policy Optimization]]).


## Takeaways
- We should have a basic grasp of policy gradients, how they're derived, and common variants of policy gradients used by populate RL algorithms.
![[Pasted image 20240209223441.png]]
- **==Policy optimization==** aims to learn a policy that maximizes the expected return over sampled trajectories. To learn this policy, we can use common gradient-based optimization algorithms, such as gradient ascent. Doing this requires that we (approximately) compute the gradient of the expected return with respect to the current policy (==the policy gradient==).
![[Pasted image 20240209223538.png]]
- **Basic policy gradient:** The most simple formulation of the policy gradient is shown above; We can compute this expression in practice by taking a sample mean over trajectories that are gathered from the environment by acting according to our current policy.
- - **Policy Gradient variants**
	- The simplest variant of the policy gradient requires many trajectories to be sampled to generate an accurate estimate of the policy gradient. To mitigate this, several variants of the policy gradient can be derived, including the ==reward-to-go== and ==baseline== policy gradients.








