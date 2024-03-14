#article 
Link: https://colah.github.io/posts/2015-09-Visual-Information/

--------

Information theory gives us precise language for describing a lot of things. 
- How uncertain am I?
- How much does knowing the answer to question A tell be the answer to question B?
- How similar is one set of beliefs to another?

Information theory crystallizes vague theories about these into precise, powerful ideas.
Unfortunately, information theory can be a little intimidating, but it doesn't have to be!

# Visualizing Probability Distributions
- Before we dive into information theory, let's talk about how we can visualize simple probability distributions.
- We can visualize a single probability distribution:
![[Pasted image 20240313170239.png|50]]
- What if we want to visualize two probability distributions at the same time? It's easy if they *don't interact* (if they're statistically independent) -- we can just use one axis for one variable and one for the other!

![[Pasted image 20240313170332.png|200]]

- Notice that straight and vertical lines that go all the way through -- that's what *independence* looks like! 
	- The probability that we're wearing a coat *doesn't change* in response to the fact that it'll be raining next week!

- When variables interact, there's *extra* probability for particular pairs of variables, and *missing/reduced* probability for others!
	- If we consider {weather today} and {clothing today}, then there's "extra probability mass" that I'm wearing a coat and it's raining, because the variables are correlated, making eachother more likely. It's more likely that I'm wearing a coat on a day that it rains than the probability of coat+sunshine.

![[Pasted image 20240313170727.png|300]]

While this might *look* cool, it isn't very useful for understanding what's actually going on.

Instead, let's focus on one variable like the weather!
- We know how probable it is that it's sunny or raining.
	- For both cases, we can look at the *conditional probabilities* -- how likely am I to wear a T-shirt *if it's sunny?* How like am I to wear a coat *if it's raining?*

![[Pasted image 20240313171240.png]]

There's a 25% chance that it's raining. *If it's raining*, there's then a 75% chance that I'd wear a coat! So the probability that it's raining *and I'm wearing a coat* is 25% *times* 75%, which is approximately 19%.

We can write this as:

$p(rain, coat) = p(rain) * p(coat | rain)$ 

This is a single case of one of the most fundamental ideas of probability theory!

$p(x,y) = p(x) * p(y|x)$      (which can also be written as)   $p(x,y) = p(y)*p(x|y)$ 

Above, we're *factoring* the distribution, breaking it down into the product of two pieces.



