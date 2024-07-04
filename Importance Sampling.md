References:
- [Video: Mutual Information's Importance Sampling Introduction](https://youtu.be/C3p2wI4RAi8?si=HxU_JpVMBycDI_vP)
- Video: [Mutual Information's Monte Carlo and Off-Policy Methods (24:00)](https://www.youtube.com/watch?v=bpUszPiWM7o)

Define: *Proposal Distribution*

Enables sampling an expectation of one distribution, using samples from another.
Useful in the context of [[Off-Policy]] learning, where we have a *behavior policy* that's exploring the MDP, but we have a *target* policy that's of interest to us.

![[Pasted image 20240626153912.png|300]]
Above: "b" is the behavior policy. 
The thing we're after is the expectation under b of the return... multiplied by the ratio of the probabilities of that return under the two policies. This ratio is called *rho*, and it turns out that it's equal to the product of the ratios of the two policies over the remainder of the trajectory.
We also require coverage, meaning if the target policy has a non-zero probability of taking some action in a specific state, then so must the behavior policy. $\pi(a|s) > 0 \implies b(a|s) >0$. This ensures that in the limit of infinite data, we won't have zero data in the places where the target ends up.


