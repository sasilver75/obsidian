References:
- [Mutual Information Function Approximation](https://youtu.be/Vky0WVh_FSk?si=4LDeYmjVrHzs_p50)
- [David Silver Function Approximation](https://youtu.be/UoPei5o4fps?si=zxe8AENyoNU6R43y)

Generalization: If an agent only visits a miniscule subset of all states, how can it generalize that experience to select high-reward actions over the remaining states?
- We use [[Supervised Learning]]! 
	- But there are RL-specific challenges; the supervised learning we call for RL is called Function Approximation

On-Policy Function Evaluation:
- Can we approximate the $v_{\pi}(s)$ in a case where the data is generated under a fixed policy $\pi$ and the state space $S$ is large?
- We assume that the true state value function can be approximated by a function of that state and some parameter vector w: 
	- $v_{\pi}(s) \approx \hat{v}(s,w)$ 

Let's talk about an example
- If $s$ is an image, then $S$ is the state space of possible images.
- Let's say that we have two features of $s$:
	- Each feature maps from an image to a number
		- $x_1(s)$ is the average of pixel values
		- $x_2(s)$ is the standard deviation of pixel values
	- We combine these into a vector:
$x(s) = \begin{matrix}  x_1(s) \\  x_2(s)  \end{matrix}$

![[Pasted image 20250114212842.png]]
Our value function could just be a linear one, like this!

![[Pasted image 20250114212942.png]]
Consider that updating our weights will impact multiple states, since the dimensionality of our weight vector is much smaller than the number of states.
This means that we won't be likely able to represent the true value function even with infinite data.

The best $w$ settings would minimize our "value error":

![[Pasted image 20250114213536.png]]
This is just the MSE between our predicted value function and our true value function, weighted by $\mu(s)$, which is a distribution over states, weighing how much we "care" about each state; typically chosen as the proportion of time that our model stays in each state.
- "Whoah, you want us to sum over many states (which we might not be able to), and you want us to use the true value function (which we don't have) to compare again our predicted one?"
	- Right, we might not be able to calculate this, but we can still develop algorithms that can approximately optimize it, using [[Stochastic Gradient Descent]]!

Assumptions:
- States are visited in proportion to $\mu(\cdot)$ 
- Our value function $\hat{v}(s,w)$ is differentiable with respect to w, which is required for calculating gradients.
- Since we don't have access to the true value, assume we have some suitable surrogate for $v_{\pi}(S_t)$, which we'll call the target $U_t$.

A simple update rule to apply to W for each visit to a state:
$w \leftarrow w + \alpha[U_t - \hat{v}(s_t,w)] \nabla \hat{v}(S_t,w)$ 
- Note that this look a little different than the gradient descent that we're use to.... in standard supervised learning, we can directly 
- We're nudging our weights in a way that minimizesour error from the target $U$

If our target is Unbiased, meaning $\mathbb{E}[U_t|S_t=s] = v_\pi(s)$ 
- Then w will converge to a local optimum of the Value Error $\bar{VE}$ 
That's nice!
- Remember our value function is any differentiable function of W, so a local optimum is the best we can get -- no one out there doing ML is finding global optimum

Example: ==Gradient Monte Carlo==

Where $U_t$ = $G_t$, i.e. the target is just the return
But MC has tis drawbakcs - waiting till the end of the episode can make us much slower!

Maybe we use a one-step-ahead value estimate of the target
$U_t = R_t + \gamma \hat{v}(S_t+1, w)$
1. But this is biased! Think about after we initialize $w$: Is our $\mathbb{E}[U_t] = v_{\pi}(s)$? Definitely not.
2. Our target U now depends on W, which means our update rule is not a true gradient step.
	- This means we don't get guaranteed convergence to a local optimum
	- So now what? Do it anyways
		- We do! this is called semi-gradient TD, and it ... usually works well!

![[Pasted image 20250114215238.png]]

And now, for the sake of making stronger statements, let's constrain ourselves ot having a linear value function, meaning the value function is a dot product of w with $x(s)$, where $x(s)$ is a feature vector:
![[Pasted image 20250114220154.png]]
This linearity makes our optimization convex, so our local optimum == our global optimum.

---

  


