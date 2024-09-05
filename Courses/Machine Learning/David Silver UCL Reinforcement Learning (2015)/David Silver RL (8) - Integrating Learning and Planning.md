https://www.youtube.com/watch?v=ItMutbeOHtc&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=10

Thinking about architectures for learning and planning in RL. Lots of conceptual ideas, and much less math than in the previous lectures.

Agenda:
1. Introduction
2. [[Model-Based]] RL
	- Up to now, we've mostly done [[Model-Free]] learning, where we don't try to build up an understanding of the environment.
3. Integrated Architectures
	- Can we get the best of both worlds by combining Model-Based and Model-Free methods together?
4. Simulation-Based Search
	- A class of methods that are all about planning, but the main idea is that we plan by having the agent imagining what happens in the future, and it learns from that simulated experiment how to behave better.

---

# Introduction

![[Pasted image 20240629111657.png]]
- In the last lecture, we talked about learning policies directly from experience ([[Policy Gradient]] methods), where the agent runs through an environment, and considers how to adjust its policy to do better.
- In previous lectures, we talked about learning value functions directly from experience (and then indirectly using that value function to behave better).
- In *this lecture*, we'll try to do something differently -- instead of learning pi or V/Q from experience, we'll learn a *model* directly from experience!
	- The agent starts to understand the environment of the environment;
		- How the agent moves from one state to another state
		- How rewards are disbursed in the environement.
	- "If I took this action, I'll end up here, and get this reward."
- We'll use this to plan! Image we do something like tree-search to do look-ahead, imagining what might happen in the future, to construct a value function or a policy.

A model describes, for any given environment, the agent's understanding of that enviroment
- How states transition to other states
- How states lead to rewards
Once we understand these, we should able to plan better in our environment.

==Model-Free RL==
- No Model; we mean that the agen't doesn't make any effort to represent the transition function or reward function.
- Instead, we learn a value function (and/or policy) directly from experience (using MC, TD, Policy Gradients, whatever, applied directly to the stream of experience) to estimate how much reward it can get, or which actions it should take.

==Model-Based RL==
- The agent uses its experience to construct an estimate of these transition and reward dynamics in the world.
- We then ***use that model*** to *plan* a value function and/or policy from the model.

If we knew, when taking a specific proposed action
- Where we would land
- The reward we would receive for that action
We can then consider, okay, from there, what if I then took some *following* action
- Where we would land
- The reward we would receive for that action
This helps us build look-ahead search trees, letting us perform planning.

If we have this model, we can understand how to learn a value function or policy without even directly interacting with the environment.

# Model-Based RL

![[Pasted image 20240629113043.png]]
Let's start with the (experience) part, but we could start anywhere.
- From experiences, the agent begins to learn about environment. We change our model of the environment to understand what happens when (eg) we open a door. ("Learning what we think the MDP looks like")
- Once we have a model, we use the model to plan with a look-ahead process.
- We start to have interactions with the model, which generate a value function or policy. ("Solving the MDP that we believe we're in")
- We use that value function or policy to act in the real world

![[Pasted image 20240629113120.png]]
- The clearest example would be to take a domain where learning value functions or policies are hard! In Chess, there's a huge variety of different states we can be in, in chess. If we just move one piece by one square, it can completely change you from a winning position to a losing position -- there's a very *sharp* value function. This means it's a hard problem to learn a value function or policy directly!
- In contrast, the model for chess is pretty straightforward! If we can use that model to look-ahead, we can estimate the value function by planning using something like tree-search, which is really powerful in games to construct a value function when it would be hard to *estiamte* that value function through normal means.
	- A model can be a more compact and useful representation of the information about the environment than you might access directly by a value function or policy.

One thing nice about model learning is that we can efficiently learn the model by supervised learning methods!
- I'm in a state, I take a step, and I'm in another state. By supervised learning, I can say that the input is being in state s and the output is being in state s'. We have a supervision signal that tells us the right thing; we can use supervised learning to figure out the dynamics and rewards of our model.

A lot of people talk about model uncertainty; sometimes the exploration problem in RL is taking actions that help us understand the world better -- not *just* actions that get us the maximum reward from our *current* (perhaps wrong) view of the world. The model tells us all the things we know, so if we understand our model uncertainty, we can take actions that take us to parts of the environment that we don't understand very well, if we choose to!

On the downside, we have to first learn a model, and then construct a value function. These are *two* sources of approximation error (we learn an approximate model, then construct an approximate value function from it).
If we learn an incorrect model and plan with it, we'll get an incorrect answer.


![[Pasted image 20240629114928.png]]
==So what is a model?==
- It's a representation of an MDP (state space, action space, transition probabilities, reward function) parametrized by some (eg) neural network $\eta$ .
	- We'll assume the state space and action space are known, in this course.
- ==The model will estimate the state transitions and the rewards.==
- We typically assume that there's conditional independence between state transitions and rewards.

==Q: Isn't it the same to learn the reward function as it is to learn a value function?==
A: No! In chess, what would it mean to learn the reward function? It might be something like 0 for losing a game a 1 for winning a game. WE just need to learn if we're in checkmate, we lose, and if we checkmate our opponent we win, and if it's a stalemate we draw. Whereas if we're learning a value function, we need to learn about all the states that are NOT a terminal position (when all the pieces are distributed around the board). The reward learning problem is easy because it's just going to be learning about the terminal states (in chess). The reward function is just 0,0,0,0,0 until the end of the game until you win or lose. Learning about checkmate is easy, because it's mostly about whether the king has moves left. As opposed to value function learning, which is about "What's the probability that white will win, from this complicated intermediate board position." There are situations in which planning sometimes helps, because there are some cases (in tactical cases like Chess or Go) where it's necessary to plan to figure out the situation you're in.

![[Pasted image 20240629115832.png]]
Our goal is to learn this model from experience (our usual trajectory of state, action reward, state, action, reward).
- We can transform this into a supervised learning problem!

![[Pasted image 20240629120340.png]]
Can use whatever sort of model you'd like

![[Pasted image 20240629120349.png]]
The simplest case is a table-lookup one:
- A model is an explicit representation of an MDP, and tries to estimate the transition and reward functions.
We use the counts of (s,a) to learn the probability
- The 1(...) is an indicator function telling us when we end up in a certain state, after taking an action, or taking a state action. The idea is just to use the means. It's the intuitive thing, here.
Alternatively:
- We can just remember things! We record at each time step the "record experience tuple"
- To sample from the model, we can uniformly randomly pick from the tups matching your current state and action. In expectation, these are the same thing.

![[Pasted image 20240629120902.png]]
Above: ==Table Lookup Model==
In this problem, we have two states with no discounting. 
- Started in a, got reward 0, went to B, got reward zero, episode terminated.
- Each row is an episode.
We can take this experience and build a model from this experience.
Now we just need to count:
- Every time I was in A, I transitioned to B.
- From B, on 6/8 occasions, I transitioned to the terminal state with reward 1, and 2/8 occasions, transitioned to terminal state with reward 0.

![[Pasted image 20240629121039.png]]
==Now how do we plan with a model, once we've got $\mathcal{M_{\eta}}$ ?==
- How do we solve the MDP defined by $<\mathcal{S},\mathcal{A},\mathcal{P_{\eta}, \mathcal{R_{\eta}}}>$  , where the $_{\eta}$ ones are *our* model's estimations of the environment's "true" transition and reward functions, based on experience?
We can use our favorite planning algorithm to do that!
- Recall in (eg) Value Iteration we said "We can't actually use these, because we don't have knowledge of the environment" -- but now we do!

![[Pasted image 20240629121319.png]]
The method we'll talk about most in this lecture is called ==[[Sample-Based Planning]]==, which is both on the most simple and most powerful ways to plan.
- Idea: ONLY use the model toe generate samples.
- We treat the model as if it's the real environment, and just interact with the environment to see samples of where I end up.
- Instead of knowing "75% of the time it will blow me left, and 25% of the time it will below me right," we'll just *sample something* from the environment and say "Oh, looks like I got blown right in this situation", and learn from that -- just like we do in the real world.

We can then apply our usual [[Model-Free]] RL to the samples from this trajectory!
- [[Monte-Carlo Learning]] for control
- [[SARSA]]
- [[Q-Learning]]
The idea is to use the model to generate samples, and then apply model-free-based RL to the samples.

The agent has some model in its head, it imagines what's going to happen next, and it plans by solving for its imagined world, simulating an experience. Instead of *actually walking ahead,* we imagine "what if I put my foot here, then there, then there." Maybe in that simulation we either fall over or get to the pot of gold. We solve by selecting the best found simulation.

We give up these probabilities, by we get some efficiency.
- ==Even if we have the model in our hands, sampling from it is often still a good, efficient idea that gives us an advantage because we focus on things that are more likely to happen, rather than doing a naive full-width lookahead where we even consider things that happen with very small probability.==


![[Pasted image 20240629125947.png]]
So now that we have this model from real experience, we can sample some additional experiences from this model! We can sample some trajectories from this model. (In this example, it seems the starting point is random between A and B).
- The advantage from this approach is that even though we've only seen (eg) 10 trajectories, we can sample 1,000 trajectories from our model, if we have compute!

![[Pasted image 20240629130852.png]]
Model-Based RL is only as good as the model that we've learned; we should be okay with that, though -- we're always going to be approximate when we haven't seen enough data!
- Here, we learn a model and solve for that model, as opposed to solving for a value function or policy directly. Different ways to solve the RL problem.


# Integrated Architectures
- We want something that has the advantages of both model-based and model-free RL.
![[Pasted image 20240629131231.png]]
Consider two sources of experience:
- Real experience: Sampled from the *environment*; the true MDP. Our robot interacting with the real world.
- Simulated experience: Sampled from an approximated, learned *model*; An approximated MDP. We can generate ~inf streams of experience but just imagining what might happen next.

![[Pasted image 20240629131517.png]]
What if we combine these together?

![[Pasted image 20240629131527.png]]
The ==[[Dyna]]== architecture is an older, but fundamental architecture from [[Richard Sutton]], where we learn a model from real experience, but then we use both sources of experience (real, simulated) to learn your value function or policy.
- Sometimes we should trust the real experience, sometimes we should trust the simulated experience.

![[Pasted image 20240629131751.png]]
We take our model-based RL loop from earlier, but we add in this "direct RL" arc, which says that we don't *just* learn our value function/policy from planning against the model (running imaginary trajectories, and running our policy against those trajectories), but *also* from direct experience interacting with the environment using that policy.

![[Pasted image 20240629131850.png]]
The ==[[Dyna-Q]] Algorithm== is the simplest version of Dyna.
- Let's start with some action-value function Q(s,a) and some Model
- Given a state, we plan with our table-lookup model in addition to he the real experiences
- Every real step in the world, we take an action and see where we end up, then do two things:
	- Usual Q learning/SARSA update, updating our value function Q towards our TD target after one step.
	- We also update our *model* via supervised learning a little bit towards the reward we observed and the state we observed, after one step.
- In addition to this real step of learning, we have this inner loop -- a thinking/imagination loop
	- We just imagine things -- we sample N different samples from our model. We start with some random state, and some action previously taken in that state.
	- Use the model to predict R, S' for s,a
	- We then imagine that transition; according to the model. And the we apply our Q-learning step to that transition. 
	- If we keep sampling any applying idea, we keep doing better and better by both sampling the environment and then sampling our model.


# Simulation-Based Search
Now let's back off to jus the planning problem, and really consider this idea of search combined with simulations using models, and see how we can use them to make SoTA planning methods.

Key ideas: 
1. Sampling
2. Forward Search


![[Pasted image 20240701141024.png]]Forward Search algorithms don't explore the entire state space. Here, we're saying "Actually, there's a state we care about getting the answer for, that that state is the one I'm in now!" "Now" has a special importance; if I'm climbing a mountain ,we care about the next step I need to take to be able to get to the top, rather than what I'm going to have lunch tomorrow -- that's irrelevant to me surviving.
- We do forward search, focusing on what's likely to happen next in the short-term future. We do this via lookahead. 
- We start the state we're at S_t, and build a search tree where we build search trees considering actions that we could take, and where the wind might blow us. The T states are terminating states. We build a whole search tree using our model, letting us look ahead at the whole tree of what might happen in the short term future.
- ==The idea: We don't need to solve the whole MDP -- that's a waste of resources. We just need to focus on solving for the sub-MDP that's relevant, starting from now.==

![[Pasted image 20240701141313.png]]
A forward-search paradigm that uses sample-based planning; in other words, we start from now, and we imagine what might happen next; we imagine a trajectory of experience by sampling it from our model.
- Right now I AM in this situation. In chess, I imagine what my opponent will do, and then I imagine what I'll do in response, and what they'll do in response, etc... and then I learn from that imagined experience.
- It's forward search because we're rooted in the current moment, giving us a focus on what will happen next, rather than distributing our efforts over irrelevant-to-us-now parts of the MDP.
- We end up seeing the situations that actually matter by sampling from our model+policy.

Then, once we've got these trajectories of experience, we can apply [[Model-Free]] RL to simulated episodes! We apply our familiar techniques to those trajectories (eg MC Learning, TD Learning, SARSA).

![[Pasted image 20240701141556.png]]
We start from now (s_t, for simulation k), and we generate multiple episodes of experience.
- We then apply some model-free RL to these simulated episodes, using whatever our favorite method is.
	- MC Control gives us a method called Monte-Carlo Search
	- SARSA gives us a method called TD search

![[Pasted image 20240701141719.png]]
Let's start with the simplest possible version.
- Given a model and a simulation policy (some way that we're going to pick actions in our imagination).
- We consider, from our root state, all of the actions we could take from our root state...
- For each one, we generate k episodes by sampling from our model and the simulation policy.
- We evaluate each of these (initial) actions by mean return (Monte Carlo evaluation)
	- I could do a "left" action
		- I run 100 simulations from after taking a "left" action, and then take the mean of returns
	- I could do a "right" action
		- I run 100 simulations after taking a "right" action, and then take the mean of returns from these simulations; that's our evaluation of how good it is to go right.
- We see that if we just evaluate our actions, that gives us an action value function for the root of our search tree (Q(s,a)). By the law of large numbers, we get the true value function for the simulation policy.
- All we do is pick *real actions to take* by picking the one with the highest Q(s,a).

Q: If you simulation policy is poor, it wouldn't work well, right? Do we update it as we go?
A: That's our next slide! This was "simple MC search," where we treat the policy as fixed.

Now let's talk about something that really solves SoTA problems!
[[Monte-Carlo Tree Search]] (MCTS)
![[Pasted image 20240701142547.png]]
Again we start from the root state
- Generate trajectories of experience from the root, using our current simulation policy.
- The difference is that we view this policy $\pi$ as something that's living; that can improve.
- We evaluate *every state action pair that we visit*...
	- We build a search tree containing every state we've visited so far, and all of the actions we've tried from those states so far.
- So we run a simulation out, and we continuous estimate action value for each of the intermediate (s,a) along that trajectory.
	- We again do this by taking the mean return from every point onwards.
	- So we basically record, at every part of our search tree... these Q(s,a) values, just by counting and taking the mean of the returns of times we pass through a state, action pair..
- At the end of the search, again we pick the action that has the highest Q(s,a) value at the root.
- But we can use this rich information in the search tree to make our search policy better.
![[Pasted image 20240701144912.png]]
After every simulation, we're going to make our simulation improve.
- We do this just like how we do policy improvement
- We look at Q values, and maximize over Q values in the search tree, to make them better.
- The distinction is that here we don't have a complete table of Q values everywhere -- we only have them within our search tree
Two phases
- In the tree
	- We improve the policy, picking actions to maximize the Q(S,A) we have stored in the tree.
- Beyond the tree (where we have no stored information and we haven't seen)
	- We behave according to some default, random simulation policy (could be naive)
We repeat (for each simulation):
- Evaluate states via MC evaluation
- Improve our tree policy, e.g. by epsilon-greedy over the Q values.
This is basically Monte-Carlo Control, but applied to simulated episodes of experience that start from the root state.


![[Pasted image 20240701145316.png]]
![[Pasted image 20240701145416.png]]
We define our reward function as just winning or losing the game, in the final terminal position.
We consider policies for both sides (self-play), and try to consider value functions saying: "How good is state s, under a policy?"
- We're trying to learn the optimal value function... which tries to find something like the minimax value (? We'll get more into this in the final lecture).

So *simple monte carlo search* might look like this:
- Roll out some games using the simulation policy, and average out returns to determine our Q(s,a) at our root.

We could apply [[Monte-Carlo Tree Search|MCTS]] too:
![[Pasted image 20240701145751.png]] 
- A state we're visiting for our first time (with the star)... we run some rollout until we run into our squarer state, the terminal state. 1 means that the Black team won the game.
- So we started, ran a simulation; so we can start storing some statistics in our node.


![[Pasted image 20240701145806.png]]
If we run one iteration further (starting back in our root state)... and building up our search tree of things we've visited so far. We add a new position into our search tree, and run a new simulation from there onwards. White ends up winning here instead of black, so we update our statistics in our tree.

So because it wasn't successful to take this action, maybe we try a different action (using the tree policy to guide us, see?):
![[Pasted image 20240701145907.png]]
We run a simulation from this new point, and see that black wins this time. We update the parent nodes here.

We continue
![[Pasted image 20240701145954.png]]
We see that we get richer and richer contingencies in our search tree; we start to expand our tree of lookahead possibilities towards things that are most promising. 
- We see that white wins in this situation; so we back that information up the tree.

Still, the left side looks better than the right side, so we run another simulation:
![[Pasted image 20240701150045.png]]
And start to expand the parts of the search tree that look the most promising.
- What happens is, as we run this out, we see that it very *deeply* develops the part of the search tree that it thinks are most promising, and it completely *ignores* parts of the search tree that it thinks are useless.
- But obviously we need to make sure that the algorithm spends *some time* exploring parts of the tree that are unexplored.

![[Pasted image 20240701150630.png]]
Why is MCTS a good idea?
- Every episode, we go back to the root and pick again which path to follow through the search tree. So it's a kind of best-first search
- We evaluate states dynamically, evaluating the position that we're in *right now*, instead of evaluating the entire space offline like in DP.
- Uses sampling to break the curse of dimensionality; we don't have to consider all possible things that the environment might do -- we just sample.



Although MCTS is a very effective method, it's not the only member of this family
 The key ideas are doing:
 - Forward Search
 - Sampling
If you use these together, with model-free RL methods being applied to these sampled experiences/simulations, we'll go far.


[[Temporal-Difference Search]]
![[Pasted image 20240701151036.png]]
Instead of applying MC, let's apply [[Temporal Difference Learning|TD-Learning]], or [[SARSA]] from earlier in the course!
- Whereas MCTS applies MC Control, we'll instead apply TD search to apply SARSA to the sub-MDP from now.

![[Pasted image 20240701151158.png]]
![[Pasted image 20240701151244.png]]
What does it look like?
- We start again from our real, start state (now, s_t).
- We estimate an action-value function Q(s,a); we can store it in the nodes of our search tree in the same way.
- For each step of the simulation, instead of waiting until the end of the episode and taking the real return of the whole episode, we instead estimate it via *==bootstrapping==*!
	- The reward is the return from one step, plus the predicted reward from then on.
We select actions by acting greedily to our Q values like before
==The only thing that's changed== is how we update our Q values; using TD-Learning instead of Monte-Carlo learning

We won't have time to go into this, but we *can* use function approximation for Q (for either MCL or TD-Learning).

[[Dyna-2]]
![[Pasted image 20240701151700.png]]
Last topic! Let's come back to the Dyna idea, where we don't *just* have to learn from real experience *or* simulated experience, we can combine them together! So let's do the same thing but with our forward search algorithm, now.
- We maintain two value functions, which you can think of as:
	- Long-Term Memory (updated from real experience) "In general, it's good to make X motions with your arms to climb"
	- Short-Term Memory (this is like your search tree; tells us how good it is in our search tree.. learning from simulations, to be in a certain situation) "I've done some simulations, and it looks like that rock up there is crumbly, so I don't think we should go this way"


![[Pasted image 20240701151942.png]]
The blue line is the Dyna 2 idea, which combines the best of both worlds
- Runs real trajectories, and does search over simulations.o


