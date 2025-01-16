
![[Pasted image 20250115234623.png]]
A robot iwth a camera that isn't told its exact location might be staring at a wall but still impacted by things going on behind it, meaning the markov assumption is broken.

![[Pasted image 20250115235827.png]]
Here's an example of what he's talking about -- these two states in a maze (white part is maze, black is walls) look the same to the agent, so it's not markovian.
- If hte goal was to go from the top-right to the top-left part of the maze
	- Then the bot should be going dow in the right example, and up in the left example
	- So there's no single policy that would do the right thing in both cases

So how could we construct a Markov agent state in this maze?
- Full history would be Markov, but it would be rather large
- Let's say we consider storing not just the observation we see right now, but also the previous observation; would that work? It depends on the policy and whether the state transitions are completely deterministic, or if there's some noise (and if you press down, you go up)
	- But if even then the last two states might not be Markov, depending on your maze!
- We could store our past actions

![[Pasted image 20250116000245.png]]

