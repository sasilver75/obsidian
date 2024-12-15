---
aliases:
  - Outcome Reward Model
  - ORM
  - Outcome-Supervised Reward Model
---
Reference:
- [Jonathan Berant (Tel Aviv University / Google) / Towards Robust Language Model Post-training](https://youtu.be/2AthqCX3h8U?si=KLrjbkKOuhZyIwWN)

A model that, given a prompt and a response, produces some score output that is meant to highly-correlate with human evaluator's rankings.

Training reward models from labeled human preference datasets helps us scale the (otherwise expensive) notion of "human" supervision, allowing us to fine-tune models using techniques like [[Reinforcement Learning from Human Feedback|RLHF]], where the reward signal serves as reinforcement, guiding our models to produce human-preferred outputs.

Variation/Contrast: ==[[Process Reward Model]] (PRM)== (in the context of which, normal Reward Models are called ==Outcome Reward Models== (ORM)). While outcome supervision provides feedback on a final result, process supervision provides feedback for each intermediate reasoning step.


![[Pasted image 20241215003002.png]]




From https://youtu.be/xJqCJ9QwTgU?si=e3hDyPfHltYgUUzv:
![[Pasted image 20241215140023.png]]
![[Pasted image 20241215140027.png]]
![[Pasted image 20241215140030.png]]
![[Pasted image 20241215140037.png]]
![[Pasted image 20241215140045.png]]
![[Pasted image 20241215140051.png]]
![[Pasted image 20241215140054.png]]
![[Pasted image 20241215140057.png]]
![[Pasted image 20241215140103.png]]
![[Pasted image 20241215140147.png]]
![[Pasted image 20241215140219.png]]
![[Pasted image 20241215140351.png]]
![[Pasted image 20241215140414.png]]
![[Pasted image 20241215140559.png]]
![[Pasted image 20241215140652.png]]
Positional bias of GPT-4 as a reward model. Pairwise grounding with random tie breaking made the bias much less significant. 
![[Pasted image 20241215141037.png]]


