#lecture 
Link: https://www.youtube.com/watch?v=JlK46EzImM8

Topic: Code Generation


![[Pasted image 20240416223908.png]]

![[Pasted image 20240416223856.png]]

It only makes sense to use program synthesis when it's easier than writing the program yourself.

It can be very difficult to set up all of the logical conditions (basically units tests) that need to be specified in order to get the output you want.

Another example is ==Synthesis from examples==

![[Pasted image 20240416224449.png]]

Still, you can satisfy all the given input/output examples, but still have it not be what you want.
If it seems hard to specify what you want to the model... that's because it is!

Examples are always ambiguous! there are probably an infinite number of programs that satisfy the behavior of the examples that you give -- there's an implicit human preference! Some programs are *obviously* desirable compared to others.

Part of the problem of this is the ambiguity inherent in human language
- Any word in the dictionary often has many definitions -- but humans somehow do just fine talking in this ambiguous medium!
	- ==Ambiguity isn't even a bug, it's a feature (of efficiency!)==
![[Pasted image 20240416225117.png]]
Above: [[Winograd]] Schemas


![[Pasted image 20240416225321.png]]
Above: She means the middle guy, because if she meant the right, she'd say "The one with the hat!"
==Rational Speech Acts==

![[Pasted image 20240416225751.png]]
	The Square! Because otherwise they'd say the Circle!
But a *==literal listener==* wouldn't know what to pick!


This relates to program synthesis!
- We're speaking in (eg) input output examples to a synthesizer of programs, which is our listener, trying to determine what program we're referring to.


OpenAI [[Codex]]
- Basically GPT3 trains on some code samples to generate code based on python dosctrings
- Authors introduced HumanEval benchmark, a manually created dataset of 164 problems.
	- Each problem ha s set of hidden tests, a program is correct if it passes all hidden tests.
- Metric: ==pass@k== -- Out of k samples, at least one passes the test
- Result: 
![[Pasted image 20240416231323.png]]


Followed by google's [[AlphaCode]]

![[Pasted image 20240416234204.png]]


Talk of [[Toolformer]] use by language models
![[Pasted image 20240416234853.png]]





