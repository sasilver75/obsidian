A technique in which a ==model is trained on examples of increasing difficulty==, where the definition of "difficulty" may be provided externally or discovered automatically as part of the training process.

The idea is that the model can learn general principles from easier examples, and gradually incorporate more complex and nuanced information as harder examples are introduced (edge cases).

A concept of "difficulty" must be defined, which may come from human annotation, an external heuristic (eg shorter sentences rather than longer ones, in language modeling), or the performance of another model.

Difficulty can be increased steadily, or in distinct epochs using a deterministic schedule. 

Curriculum Learning is sometimes combined with [[Reinforcement Learning]], such as learning a simplified version of a game first.


-----
Comparing Active Learning and Curriculum Learning

- [[Active Learning]] (selecting the most informative data to label next; minimizes the amount of data we need to collect)
- [[Curriculum Learning]] (ordering the examples in a dataset from easiest to hardest)
----
