https://www.youtube.com/watch?v=NHRoXvPaZqY&list=PLoROMvodv4rOpr_A7B9SriE_iZmkanvUg&index=6

----

Recall: Last week was about counting.
![[Pasted image 20240707134537.png]]
One of the big meta takeaway is that sometimes you get very large numbers.

![[Pasted image 20240707134655.png]]
What does Mutually-Exclusive mean?
![[Pasted image 20240707134719.png]]
Recall that Events are sets of Outcomes, within some Sample Space.
If no outcome is shared between two events, they're independent.

---

![[Pasted image 20240707135936.png]]
Sometimes it's easier to calculate the probability of some event *not happening*!
==Let's say ?=100 people== that I know

Last class we talked about sample spaces where event event is equally likely, which can help us use counting rules to attack a problem.

Flip: So what's the probability that we see *nobody* that we know in a room of 268 people?
- Can we set this up so that the probabilities are the same for every outcome?

We want to be able to use (size of event space / size of outcome space) = answer
Which is only possible when every outcome is equally likely.

Sample space size: $17000 \choose 268$ 

One sample would be a random selection of 268 people chosen (order of selection doesn't matter)

In a subset of those, we don't know anyone in the class.
How can we calculate that?

The number of ways we can choose 268 people we *don't know* is 

Event space size: $17000-100 \choose 268$

So $17000-100 \choose 268$ / $17000 \choose 268$ = ~.20 = 20% probability that we don't know anyone in the room!
- So 80% probability that we know *someone* in the room.

----

![[Pasted image 20240707141207.png]]

[[Conditional Probability]] is where probability starts to get exciting; it's how we update beliefs. 

![[Pasted image 20240707141945.png]]
When we condition on F, it's like we shrink into the world of green outcome espaces... because we've shrunken our world to the world where f has happened.
- So $P(E|F) = P(E \cap F)/P(F)$ 


![[Pasted image 20240707142239.png]]
Q: What happens if P(F) is zero? Won't P(E|F) blow up? 
A: But conditional probabilities mean "the probability of an event, given that I've *observed* some other event!" But we can't observe an event to condition of if its probability is zero. "What's the chance that E has happened, given that F has happened"
[[Chain Rule of Probability]]

![[Pasted image 20240707142725.png]]
These numbers aren't that interesting; the interesting thing is observing how the probabilities change when you observe other things!
Maybe we all care about the probability that someone watches a movie, conditioned on them watching *some other movie.*

![[Pasted image 20240707142812.png]]

![[Pasted image 20240707142839.png]]
![[Pasted image 20240707144130.png]]
![[Pasted image 20240707144451.png]]

----

![[Pasted image 20240707144536.png]]
You can't calculate the probability of the baby crying. What information would you need?
- The probability that a baby cries given that she hasn't pooped

![[Pasted image 20240707144715.png]]
![[Pasted image 20240707144921.png]]
The [[Law of Total Probability]]
![[Pasted image 20240707145803.png]]

----

# Bayes Theorem


![[Pasted image 20240707150051.png]]


![[Pasted image 20240707150457.png]]

![[Pasted image 20240707150656.png]]
It really just falls out of the chain rule of probability

![[Pasted image 20240707150817.png]]
Example: It's easy for us to determine P(E|F) by looking at a collection of spam emails, but we really want to be able to use the opposite: What's the probability of spam, given some word use?

60% of all email is spam ==P(H) = .6==
20% of spam has the word "dear": ==P(E|H) = P("dear" | spam) = .20== 
1% of non-spam has the word "Dear" P(E|H^C) = .01

You get an email with the word "Dear" in it -- what's the probability of it being spam?

P(spam | "Dear") = (P("Dear" | spam) x P(spam)) / P("Dear")
- P("Dear" | spam) = .2
- P(spam) .6
- P("Dear") = P("Dear", Spam) + P("Dear", NotSpam)
	- P("Dear", Spam) = P("Dear" | Spam)P(Spam) = .2x.6=.12
	- P("Dear", NotSpam) = P("Dear" | NotSpam)P(NotSpam) = .01x.4 = .04
	- =.16
Plugging in: (.2 x .6)/.16 = .75

![[Pasted image 20240707152119.png]]
- If someone has SARS, there's a 98% chance it tells you that you have SARS, and a 2% chance it tells you that you don't have SARS.
- Probability that it says you have SARS given that you don't have SARS is 1%

P(Sars | PositiveTest) = P(Positive Test | Sars) P(Sars) / P(Positive Test | Sars)P(Sars) + P(Positive Test | NoSars)P(NoSars)
= .098x.005 / (.98x.005 + .01x.995)
=.330

![[Pasted image 20240707153308.png]]
So even with a "98% recall" test, the probability that you actually have SARS is .330

P(F|E) = P(E|F)P(F) / P(E)
= P(Positive Test | Sars)P(Sars) / P(Positive Test)
= .98x.005 / P(Positive Test)
- P(Positive Test) = P(PositiveTest|Sars)P(Sars) + P(PositiveTest|NoSars)P(NoSars)
	- = .98 x .005 + .01 x .995
	- = 0.01485
= .98*.005/0.01485
= 0.32996633
= 32.996%


![[Pasted image 20240707153921.png]]
P(KnowsConcepts) = .75
	P(DoesntKnowConcept) = .25
P(Correct | DoesntKnowConcept) .25
	P(Incorrect | DoesntKnowConcept) = .75
P(Incorrect | KnowsConcept) = .1
P(Correct | KnowsConcept) = .9

P(KnowConcept|Correct)?
P(KnowConcept|Correct) = P(Correct | KnowConcept)P(KnowConcept) / P(Correct)
- P(Correct | KnowsConcept) = .9 (By complement of given) 
- P(KnowConcept) = .75 (Given)
- P(Correct) = 
	- Not given, but we can derive with rule of total probability
	- = P(Correct|DKC)P(DKC) + P(Correct|KC)P(KC)
	- = .25 x .75 + .9 x .75
	- = 0.8625
= .9 x .75 / .8625
= 0.7826086957
= 78.26%