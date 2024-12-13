References:
- [Hitchhiker's Guide to Reasoning](https://youtu.be/S5l5OvJ01ws?si=N93dg0JJ1_21zjVI)

The process of drawing ==conclusions== by generating ==inferences== from ==observations.==

![[Pasted image 20241212204522.png|400]]
![[Pasted image 20241212204541.png]]
You can think about it as a generative model that wants to generate inferences and conclusions from observations. You can break that down as generating inferences from data, and then generating a conclusion from the observations and inferences.

Correctness:
![[Pasted image 20241212204617.png]]
Can think about this as an expectations over inference steps... We integrate over lambda, our inference steps... We weight our guy by the probability of an inference step, given an observation. That probability of generating an inference given an observation is the key thing that we want to learn to generate in our reasoning models. 

Crudely: Maximize the probability of being correct, given the observations.

All reasoning research... basically boils down to how do you generate these inference steps, (lambdas). Whether it's [[Chain of Thought|CoT]], [[Monte-Carlo Tree Search|MCTS]], it's all about this central question of how to generate observations

![[Pasted image 20241212215654.png]]




