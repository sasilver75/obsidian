References: 
- [StatQuest Entropy Clearly Explained](https://www.youtube.com/watch?v=YtebGVx-Fxw&t=1s)
- [[Chris Olah]]'s fantastic [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) post

The expected surprisal of an event!

----
Constraints on "surprise":
1. An event with probability 1 has no surprise
2. Lower-probability events are strictly more surprising
3. Two independent events are exactly as surprising as the sum of those events' surprisal when independently measured
----

![[Pasted image 20240422155623.png]]

![[Pasted image 20240629001745.png]]
It's the number of bits required to encode some random variable X. The uncertainty over what value the random variable is going to take on.
- It's a probability-weighted sum of the $log_2$ (measuring in bits) inverse probability; 
- If I had a distribution over values the random variable can take home... values that are very likely, I want to encode with a small number of bits. Values that are unlikely can use a larger number of bits, given a number of bits that I have to encode samples from a random variable.