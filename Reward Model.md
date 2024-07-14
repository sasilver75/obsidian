---
aliases:
  - Outcome Reward Model
---
A model that, given a prompt and a response, produces some score output that is meant to highly-correlate with human evaluator's rankings.

Training reward models from labeled human preference datasets helps us scale the (otherwise expensive) notion of "human" supervision, allowing us to fine-tune models using techniques like [[Reinforcement Learning from Human Feedback|RLHF]], where the reward signal serves as reinforcement, guiding our models to produce human-preferred outputs.

Variation/Contrast: ==[[Process Reward Model]] (PRM)== (in the context of which, normal Reward Models are called Outcome Reward Models). While outcome supervision provides feedback on a final result, process supervision provides feedback for each intermediate reasoning step.