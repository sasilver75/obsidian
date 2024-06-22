https://youtu.be/2pWv7GOvuf0?si=tqABzc5Z8nAKSHFr

A course co-offered by Deepmind and UCL
- This is a 9 year old course, but it's one that's highly recommended... It's only 10 lectures!

-----

What differentiates RL from Supervised/Unsupervised learning?
- There's no supervisor (no one tells us the right action to take), just a sparse reward signal (5 points, 2 points, 10 points).
- Feedback is delayed, not instantaneous. There's a credit assignment problem about hi.
- Time really matters -- we care about sequential decision processes -- not i.i.d. data. Here, we have a dynamic system, with an agent moving through a world -- what that agent sees at one second may be very correlated with what it sees at the next second.
- In RL, the agent gets to take actions to influence its environment, which affects the subsequent data it receives!

Examples of RL problems
- Flying a stunt maneuver in a helicopter
- Playing backgammon or Go better than humans
- Managing an investment portfolio
- Controlling a power station
- Making a humanoid robot walk
- Make a single program play a suite of games


Rewards
![[Pasted image 20240621011357.png|300]]
- A reward is just a scalar feedback signal R_t, where at every timestep t we define this feedback signal indicating how well the agent is doing at step t.
	- Does this cover what we mean by goals in all problems?
	- RL is based on the following premise: ==ALL GOALS can be described by the maximization of SOME expected cumulative reward, and the maximization of that through time, into the future.==
Thus, SHAPING your reward is highly important!

Question: What if it's a time-based goal, like doing something as fast as possible?
- A: We define the reward signal to be -1 per timestep, and then there's some actual positive reward signals sprinkled around; so maximizing the reward means minimizing time spent.


![[Pasted image 20240621011809.png]]
Examples of different problems that we can create reward signals for.
- The goal is to be able to address all sorts of different problems using the same formalism; the first step is understanding that we're getting a (possibly zero) reward signal at each timestep.

So... what is this framework that can solve all these different proems?
- The goal is to ==select actions so as to maximize the future reward!==
- Actions may have some long-term consequence, and the reward might come at some future step!
	- This might mean that we have to give up some small reward now, to get a greater reward later!
		- (eg in finance) spending some money now to get more money back later
		- (eg in helicopter) running low on fuel? Stop doing backflips to refuel so you don't crash!

![[Pasted image 20240621012136.png]]
This is all our agent sees!
- OBservations of the world
- Reward coming in
Now... make a decision! We want to learn the brain

![[Pasted image 20240621012312.png]]
But this is the full picture!

Question: Do we have to limit ourselves to scalar rewards? What if my action makes my boss like me but my girlfriend hate me?
Answer: A scalar feedback signal is sufficient to describe everything we need by goals. If we have conflicting goals, the RL view of that is that the agent has to pick actions, and to pick actions, you have to be able to weigh out these out... so there must be a conversion to a scalar thing that you can decide over, and so a scalar reward signal is enough.


The ==History== is the sequence of observations, actions, rewards that an agent sees.
H_t = A_1, O_1, R_1, ..., A_t, O_t, R_t
- It's ALL the observable variables (to the agent) up to time t
Our goal is to learn a mapping from this History to next actions!

![[Pasted image 20240621012832.png]]
The history isn't that useful, because we want agentes with long lives that can do micro-second decision-making, etc... So typically we talk about ==State==, which is a summary of the History used to decide what to do next.
- ==State== is *any function of the history*
	- It could be look at the last state
	- It could be look at the last four states

Definitions of State
- ==Environment State==
	- Information used in the environment to determine what happens next (next observation, reward). Informally, if you were to say "what state is the environment in," it's that set of information.
	- This is usually NOT VISIBLE to the agent (who only sees some local, biased, masked version of the environment) -- it might be a good thing even for the agent to have its own subjective understanding of state.
- ==Agent State==
	- The agent's internal representation, which it uses to pick the next action to take.
	- This is something that *we, the developers* have a lot of control over.
	- This state can be any function of history.
- ==Information State (aka Markov State)==
	- Contains all *useful* information from the history. It's an information-theoretic concept.
	- A state S_t is Markov if any only if
		- P(S_t+1|S_t) = P(S_t+1|S_1, ..., S_t)
		- Basically says the probability of the next state, given your current state... is the same as if you considered all previous states. So our current state has all the information we need.
		- "The future is independent of the past, given the present."
	- If we retain everything, that's also by definition a Markov state. The environment state is also always Markov.

![[Pasted image 20240621013328.png]]
![[Pasted image 20240621013319.png]]
![[Pasted image 20240621013630.png]]

Question about multi-agent RL
Answer: From the perspective of each agent, they can consider the other agents to be part of the environment -- it doesn't have to change the formalism. But the rest is beyond teh scope of this course.


![[Pasted image 20240621014443.png]]


==Fully Observable== (Agent can see all of environment) vs ==Partially Observable== (Robot with a camera, poker agent where you can't see other hands)


Let's now open up an RL agent and see what's inside!
- We've talked about the problem, but haven't talked about what it means to solve this problem -- what are the parts of an RL agent? 
	- This will help us build a useful vocabulary/taxonomy for talking about these problems.

Non-exhaustively, they are:
- ==Policy==: How the agent chooses its actions to take.
- ==Value Function==: Determines how good it is to be in a specific state; how much reward do we expect to get if we take a specific action and land in a specific state?
- ==Model==: How the *agent* thinks the environment works -- the agent's representation of the environment.

==[[Policy]]==
- A map from state to actions
- We could have a deterministic policy $a = \pi(s)$
	- We want to learn this thing from experience such that the policy guides us to getting the maximum possible reward (the RL goal!)
- We could have a stochastic policy (which might be useful to make random, exploratory decisions): $\pi(a|s) = \mathcal{P}[A=a|S=s]$   ... where $\mathcal{P}$ is a probability distribution

==[[Value Function]]== 
- A prediction of expected future reward
- We need this to learn to choose between (state1 and state2) or (action1 or action2)... and we do that by thinking about the expected total future reward.
- $v_{\pi}(s) = \mathcal{E}_{\pi}[R_t + \gamma R_{t+1} + \gamma^2R_{t+2} + ...]$  where \gamma is some sort of discounting going into the future, which says we care more about immediate rewards than later rewards.
	- If we cane compare different future states using a value function, we make optimize our behavior to optimize reward!

==Model==
- A model predicts what the environment will do next, which can be helpful in determining what to do next
- Two parts to the mode:
	- ==Transition Model==: Predicts the dynamics of the environment; $\mathcal{P}$ predicts the next state (i.e. dynamics). If the helicopter is *here* and does *this*, then it will likely be *there.*
	- ==Reward Model==: Predicts the next (*immediate*) reward with $\mathcal{R}$. If the helicopter is in *this* situation, then it will get 1 reward for staying alive.
![[Pasted image 20240622142538.png|300]]
- A lot of the course will focus on [[Model-Free]] methods that don't use a model at all! It's not a requirement/necessary to explicitly model the environment like this, but you can!

Categorizing RL Agents: We can taxonomize agents by the three components above!
- ==Value-Based==
	- Has a Value Function
	- No Policy (Implicit; it just has to look at the Value Function and pick the best action greedily)
- ==Policy-Based== Agent
	- Instead of representing, inside the agent, the Value Function, how well it's going to do  from each of these states, instead they explicitly represent the Policy!
	- A policy-based agent maintains some kind of data structure telling us the predicted action from any state.
- ==Actor-Critic== Agent 
	- Policy
	- Value Function
	- Basically combines the two above together and tries to get the best of both worlds.
- ==[[Model-Free]]==
	- We don't try to explicitly understand the environment; we don't understand the dynamics of how the environment behaves.
	- Instead, we go directly to the policy or value function; we see experience and figure out a policy how to behave to best get a reward.
		- (Question: How do you even do this without building a representation of the environment first?)
	- Policy and/or Value Function
	- No Model
- ==[[Model-Based]]==
	- First, we build a model of how our environment works (dynamics of a helicopter), and then figure out how to behave based on this.
	- Policy and/or Value Function
	- Model

![[Pasted image 20240622144559.png]]
We can either have a value function or not, a policy or not, and a model or not.
- Ultimately, the agent has to select actions, and to select actions, it either needs to have a policy to pick it, or a value function to implicitly give us a policy where we greedily follow the value function through states.

## Problems within Reinforcement Learning

Learning and Planning
- Two fundamental problems in sequential decision making
	- ==Reinforcement Learning== problem:
		- The environment is initially unknown! We drop our robot onto the factory floor and tell it to get maximum reward. We don't tell ti how the factory operates or how the wind blows, it figures out how what it has to do (given the environment and objectives) through interacting with the environment through trial-and-errror learning.
		- The environment is initially unknown
		- The agent interacts with the environment
		- The agent improves its planning
	- ==Planning== problem:
		- A model of the environment is know (dynamics of the wind, differential equations describing how helicopter moves, etc).
		- Instead of interacting with the environment, it can perform internal computations using its perfect model of reality WITHOUT any external interactions.
		- As a result of this, it improves its policy 
		- AKA deliberation, reasoning, introspection, pondering, though, search


![[Pasted image 20240622145014.png]]

![[Pasted image 20240622145053.png]]
Question: If you have a bunch of actuators, doesn't the branching factor of the search place explores!
Answer: Yes, there are tactics for this we'll talk about later.


Exploration and Exploitation
- Reinforcement learning is like trial-and-error learning; we don't know what the environment looks like -- we have to figure out through trial and error to figure out which parts of the space are good and which parts are bad -- but we might be learning rewards along to way!
- So we want to learn a good policy from its experiences of the environment, without losing too much reward along the way by exploring.
	- It has to balance exploration vs exploitation
		- ==Exploration==: Finds more information about the environment
		- ==Exploitation==: Maximizes reward given what's currently known about the environment
![[Pasted image 20240622145621.png]]

Prediction: Evaluate the future
- How well will I do if I follow my current policy?
Control: Optimize the future
- What's the optimal policy? 

Typically we need to be able to solve the prediction problem first before solving the control problem.

![[Pasted image 20240622150936.png]]
THE COURSE GOING FORWARD!



