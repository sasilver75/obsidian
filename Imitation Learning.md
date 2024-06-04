In Imitation Learning, instead of trying to learn from the sparse rewards or manually specify a reward function, an expert provides us with a set of demonstrations that serve as supervision.

The agent then tries to learn the optimal policy by following the expert's decisions.

Generally, imitation learning is useful when it's easier for an expert to demonstrate the desired behavior rather than to specify a reward function.

((In the context of language modeling, this is just supervised learning, I think, but where the dataset is constructed to enable the model to learn some specific behavior, like being a chat assistant))