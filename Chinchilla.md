March 29, 2022
Paper: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

Answers the question: "With a given compute budget, and ignoring inference costs, how do we choose between the number of parameters of our model and the number of tokens we train on?"

==It can make sense to train a model that's *smaller* than Chinchilla optimal==, and train it for *longer* than Chinchilla would tell us, because if we're going to deploy the model at mass scale, we care *much more* about inference cost than we do training cost! - [Link](https://finbarr.ca/llms-not-trained-enough/)

 Results in scaling laws that have parameters and ==tokens linearly increase at a 20:1 token to parameter ratio==."

---
April 17, 2024 Update
People calling the replication of the paper into question: 
- (Jan 21, 2023): https://x.com/suchenzang/status/1616752482226671620
- (April 17, 2024): https://x.com/tamaybes/status/1780639257389904013

---