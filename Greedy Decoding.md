Langauge models generate a probability distribution over the next token.
The "decoding" process refers to the manner in which we choose to sample from this distribution.
"Greedy decoding" means that we always choose the token with the most probability mass.

Note: Decoding strategies are orthogonal to that of [[Temperature]], which influences the shape of the upstream probability distribution.