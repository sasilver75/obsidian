

How do we evaluate policies?
- This incorporates computing the state-value function $v_{\pi}(s)$ or action-value function $q_{\pi}(s,a)$ for a given policy, for all state/action pairs.
- ((Not good)): Exploring the MDP means pass over all states (and actions?) in what is called a *sweep*, and we stop doing sweeps once the state-value function stops changing by more than some specified amount.

Policy Improvement
- Given some (say, deterministic) policy $\pi$, how do we determine a better policy?
- For an optimal policy $\pi_*$, we know that it must select the action with the maximum action-value: $\pi_*(s) = argmax_aq_*(s,a)$ 
- We define a *new policy* $\pi'$ which always selects the max action-value in every state. We say the new policy is acting *greedily* with respect to the action-value function... 
- Because of something called the Policy Improvement Theorem, we're guaranteed that the new policy is at least as good as the policy of the given value function... so this always helps.
![[Pasted image 20240623203353.png|300]]