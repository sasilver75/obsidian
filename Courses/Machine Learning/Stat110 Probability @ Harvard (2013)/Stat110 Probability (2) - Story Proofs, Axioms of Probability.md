https://www.youtube.com/watch?v=FJd_1H3rZGg&list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo&index=2

---

Recall:
![[Pasted image 20240703190027.png]]

If I have 10 people, and I want to split them into a team of 6 and a team of 4, how many ways are there to do that?
- $10 \choose 4$ 
- Alternatively, $10 \choose 6$
In either case, the remainder is the team of 6 (or 4)
We actually just proved that $10 \choose 4$  = $10 \choose 6$  (because the denominators are 6!4! and 4!6! respectively)

On the other hand, if we wanted to select two teams of 5
- Then if we just did $10 \choose 5$ then we're off!
If we assume teams are 1-5 and 6-10, then there's only one way to do that
In this case, we're not designating some distinction between the two teams (ie there's not a red team and a blue team, there are just two teams)
So in this case it's $10 \choose 5$/2 , because we're double-counting...
There's a clear difference between a team of 4 and a team of 6, but two teams of 5... unless they have different jerseys, it's equivalent!
==The above is a subtle distinction that gets missed==
- It's a little too simplistic to say "order matters or order doesn't matter"; think correctly!


![[Pasted image 20240703190536.png]]
Last time we mentioned that 3/4 of these entries were "obvious" from the multiplication rule; let's talk about the fourth one, the top-right one.

So where does it come from?

Problem: 
- We want to pick k times from a set of n objects.
- Order doesn't matter, and we're sampling with replacement.

Last time we stated it was $n+k-1 \choose k$ ... but why? Let's get some intuition for it.

Let's check some extreme cases:
- Where k=0 (meaning you don't pick anything): $n-1 \choose 0$ is always 1. (Anything choose zero is always 1; you just don't choose)
- Where k=1 (meaning just pick one): $n \choose 1$ is always n. Note that if we pick once, there's no difference if there's replacement or not.
- Where n=2: $2+k-1 \choose k$ 

