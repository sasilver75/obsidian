


Two concepts:
- Formal language grammar: A set of strings
	- This is sort of related to the set of sentences that a native speaker would accept as being ... gramattical.
- Grammar in a linguistics sense:
	- Natural langauge sentences, generally.


Deterministic Finite State Automata:
- The transition function has a transition for every possible "action"... tells you exactly where to go.

N DFA (NFA)
- 2^Q is the set of all subsets
- You could have no place to go from a given action, or multiple places to go for a given action...

![[Pasted image 20241015153256.png]]
We worked this out previously

Remark:
![[Pasted image 20241015153312.png]]
For every NFA, there exists a DFA such that the languages accepted by each are the same
- Isn't it really hard to run an NFA, since you don't know where to go?
	- But for a DFA, you know exactly where to go
- This theorem actually takes an arbitrary NFA and constructs a DFA. It's a proof by construction, so you can use this to create a DFA version of an NFA, which is easier to actually run.

What's a ==REgular Langauge==?
- Any language that I can construct either a DFA or NFA for (recall: they're equivalent).



![[Pasted image 20241015153652.png]]
A ==regular expression== is a way of describing a regular language that is accepted by an FSA
- For anything that you have an NFA/DFA, I can then construct the Regular Expression.
- Then go the other way, and say for any Regular Expression, I can construct a DFA.


But first let's do the NFA/DFA proof first.
- We'll do it by induction
![[Pasted image 20241015153902.png]]
We prove an infinite set of statements, and do it incrementally by showing that it's true for one, and showing that if it's true  for all the ones below, then it's also true for the next ones.

- Given an NFA
- We can construct a DFA such that L(A) = L(A')
- An a DFA is already an NFA, so we don't need to prove the reverse transformation.

To construct a DFA, we're going to use a trick:
![[Pasted image 20241015154132.png]]
1. Make the vocabulary the same
2. The states are going to be *sets* of NFA states
3. ... (zzz)

![[Pasted image 20241015154324.png]]
Turing an NFA into a DFA: Crazy transition function that basically uses all of them... Keeps track of all the possible states that I could be in, and writes a new transition function.... between sets of states.

![[Pasted image 20241015154431.png]]
If I'm in the empty set state, then there's no where that  I can go... "Failure state"
Then if I'm *in* a state, then I can get to all these other states. For every single q, I can transition to any of the possibilities... Recall that this is going to give us a union over all the sets we can get to.

... cool


So let's do this one as an example.
![[Pasted image 20241015155229.png]]
For states, we're gonna have:
{qo}
{empty set}
{q1}
{qo, q1}

When I apply q_0({q_0}, a)
This is going to be the union of reachable sets from here...
= {{q_0, q_1}}
So that gives us just a direct transition from {q_0} to {q_0, q_1}

We're taking an NFA and constructing a DFA.

NFA (left), DFA (right)
![[Pasted image 20241015160409.png]]
{q_0, q_1} represents the superposition of two ground-truth states that can be reached when pressing a from {q0}
If I'm in {q0}, what happens if we press b? delta({q_0}, b) = ?
If we're in q_0 and  get a b, we have to stay in q0
So then on our NFA, we look a b from {q_0} back to {q_0}
What about {q_1}?
If we press a or b, we still say in q_1, so {q_1} has a, b looping onto itself.
If I'm in the empty set state... there's no where I can go... it's just the empty set, we stay in the empty set regardless of what we press.
If I'm in {q_0, q_1}
- (In the NFA) If I'm in q_0 and do an a, you can end up at either q_0 or q_1
- (In the NFA) If I'm in q_0 and do a b, I can end up in either q_0 or q_1
- The union of these two is just the set of {q_0, q_1}, so we add a b loop back to {q_0, q_1} too.
What about the accept states?
We know in the NFA that q1 is the accept state... so should it be any state that contains it in DFA? Since that means we could have done a transition to get to that particular state! Since the DFA-ized NFA graph has the set of reachable states... and if one of those reachable states is our final state, then taking the action that led to that state could land us in a final state.

----
Break

----
For DFAs
![[Pasted image 20241015162618.png]]
The set of strings, such taht the final output state... when we apply it to W, is in the accept states....
![[Pasted image 20241015162604.png]]


And for NFAs
![[Pasted image 20241015162537.png]]
![[Pasted image 20241015162556.png]]



We want ot provde that the language that the DFA accepts is the same lagnauge that the NFA accepts

Starting with a lemma:
![[Pasted image 20241015162752.png]]
d' is a single state
d is a set of states 

We want to prove... this induction on |w|
In induction proofs, we start with the base case.
If w is the empty string, then we apply q_0 


![[Pasted image 20241015163854.png]]
NFA with epsilons
So this could accept the empty string.

So let's extend our NFAs to allow Epsilon-Transitions too!
we want to convince you that this isn't any more apowerful than a regular NFA...
We allow ourselves to have an extra symbol, Epsilon, in the delta trasition function. THat's theo nly chagne.
![[Pasted image 20241015163942.png]]
So we just need to show that it's the same 

![[Pasted image 20241015163952.png]]


---

What are the operations for a regular expression, agai? There's concatenation, or, kleen star, 

