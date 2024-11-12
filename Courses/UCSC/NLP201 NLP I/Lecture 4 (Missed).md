
Starting at slide 30 on https://canvas.ucsc.edu/courses/76196/files/9606410?wrap=1

==DFA==: ==Deterministic Finite Automata==
- 5-tuple M=<Q, Sigma, delta, q_0, F>
	- Q = finite set of states
	- Sigma = finite alphabet
	- delta = Q x Sigma -> Q, a transition function 
	- q_0 belongs to Q: The start (initial) state
	- F subset of Q is the set of final (accept) states

L(M) is a subset of Sigma*

Is the LANGUAGE of M, i.e. the set of strings that M accepts

![[Pasted image 20241010155252.png]]
![[Pasted image 20241010155301.png]]
![[Pasted image 20241010155306.png]]
The edges are the transition functions
(If you're at q_0 and see a 0, go back to q_0)

Note: I think they're deterministic because at a given state, there's only one edge for a given observation... ie from q_0, you can only see 0 and 1, and if you see 1, there's like a single thing that you should transition to.

M (our DFA with alphabet Sigma) accepts an input sequence w (w1,w2...wn, where each w_n is in sigma).

M (our DFA) accepts w (our sequence) if we start in q_0 and then reading it w (our input sequence) ends in an accept state... 
Meaning if there's a sequence of states s0, s1, ..., sn that
s0 = q0
si = d(si-1, w_i)  < plug your current state and observed word /string into your transition function to get your new state

s_n (terminal state) is a member of F (the set of final accept states)

In many problems in NLP, we want to analyze a string in terms of underlying operations -- with FSM, this corresponds to tquestion of:
- What was the sequence of states that were taken to produce some w?
	- (Producing a w, huh?)

![[Pasted image 20241010160225.png]]
a^2n represents strings of the letter a, where the length is any even number
- aa (n=1)
- aaaa (n=2)
- aaaaaa (n=3)

See that our acceptance state is the one on the right
We start at q_0
If we see two a's, we transition to the accepted state
If we see three a's, we would transition to a non-accepted state

![[Pasted image 20241010160704.png]]


Now, let's talk about ==Non-Deterministic Finite State Automata== (==NFA==)

![[Pasted image 20241010160823.png]]
See that it's the same, but we've replaced the transition function with a new one! :) 
Now, when you're in a state and you see ... 

![[Pasted image 20241010161709.png]]


![[Pasted image 20241010161833.png]]
A DFA is required to have a ==COMPLETE transition function== (I think this just means that there's only a a single transition for a given word, from a state?)
- It ACTUALLY means: For every state in the automaton, for every symbol in the input alphabet, there must be exactly one defined transition.
- ==For each state, a DFA has EXACTLY ONE outgoing edge for every symbol in SIGMA==
An NFA can have ==INCOMPLETE transition functions== .... 
- There might be states where transitions for some input symbols are ==not defined==.
- There could be ==multiple transitions== for the same input symbol
- There could be ==epsilon transitions==, allowing state changes without consuming an input symbol

L(M) = {a,b,a}
Recall that L(M) is the LANGUAGE of M (a finite automata), meaning the set of strings that M accepts.
So this accepts the string string ABA only.
Using an NFA, this is easy to specify! Because for a given state (eg q1) we don't need to spefiy both a and b, we can just specify b as a transition.

![[Pasted image 20241010163252.png]]



![[Pasted image 20241010165253.png]]


![[Pasted image 20241010165300.png]]

a\*b\*
e.g. aabb or aaaabbbb
For a DFA you can't draw a DFA to validate that
- Because every single cell has to have every single symbol (a, b)
- And so you can't keep track of the number of a's you've seen, as you parse across

Whereas for a NFA you can 

![[Pasted image 20241010165800.png]]


![[Pasted image 20241010165937.png|400]]
Remember: Sigma is some finite Alphabet


![[Pasted image 20241010165950.png|400]]

![[Pasted image 20241010171058.png]]

![[Pasted image 20241010171537.png]]
Every NFA can be converted into a DFA
![[Pasted image 20241010171612.png]]








