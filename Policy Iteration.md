The first step of Policy Iteration is Policy Evaluation, where we fix a policy and determine what the state-value function V of the policy is.
![[Pasted image 20240629000524.png|400]]
After convergence, for all $s \in S$ we have an estimation of our policy's associated state-value function.
![[Pasted image 20240629000840.png]]
Equation 2 is correct for computing the value of a *stochastic* policy.

In Policy Iteration
- We have a policy
- We evaluate it
- We use the evaluation to make an improvement to the policy
- We evaluate the new policy
- ... (until convergence)

![[Pasted image 20240629001042.png]]
The new policy gets to maximize over actions. We look at the action in state $s$ that maximizes, averaged over s' stages, the reward we get immediately, and the discounted sum of future rewards we'd get if we followed policy $\pi_k$ from then on, from s'.
- At (guaranteed) convergence, we have optimal policy. This converges faster than value iteration under some conditions.
![[Pasted image 20240629001407.png]]






==Policy Iteration==
- To find $\pi_*$ and $v_*$
- We start with some arbitrarily initialized policy pi_0, and then apply policy evaluation, determining its value function v_pi_o
- And then we can improve with policy improvement it to create a new policy pi_1... and then apply policy evaluation again... iteratively, until the policy/value function stop improving.
![[Pasted image 20240623203555.png|300]]
![[Pasted image 20240623203729.png|300]]


![[Pasted image 20240625232846.png]]