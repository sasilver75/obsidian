
2012 vs 2014
![[Pasted image 20240417182058.png|300]]
![[Pasted image 20240417182114.png|300]]

It's a very exciting time to be doing this right now. The industry interest makes 2012 look like small potatoes.

==Even since 2022, it feels like there's been an acceleration; Some problems that we used to focus on seem like they're less pressing. I wouldn't say that they're solved, but there's less of a focus on that -- which is exciting, because it frees us up to focus on more challenging things!==

Things that are blowing up:
- Natural language generation
- Image generation
- Code generation
- Search

Throughout this course, we'll use this as a test:
=="Which US states border no US states?"==
- It's a simple question, but can be hard for our language technologies because of that negation there."

1980: Chat80 was an incredible system that could answer complex questions like: "Which country bordering the Med borders a country that's bordered bhy a country whose population exceeds that of India", but if you asked it our highlighted question above, it would say "I don't understand!"
- Things that fell outside of its capacity, it would just fall flat!

2009: Wolfram Alpha hit the scene; meant to be a revolutionary language technology. To our amazement, it still gives the following behavior when we used our highlighted question: {returns a list of all of the US state} -- fail!

2020: Ada, the first of the OpenAI models, when asked it:
- "The answer is: No {begins babbling}"

2020: Babbage, later OpenAI model:
- "The US states border no US states. {babbling}"

2021: Curie, later OpenAI model:
- "A. Alaska, Hawaii and Puerto Rico
  B. Alaska, Hawaii, and the US Virgin Islands
  C. ..."

2022: Davinci-Instruct beta (It has *instruct* in the name)
- "Alaska and Hawaii"

2022: Text-Davinci-001
- "Alaska and Hawaii are the only US states that border no other US states"

A microcosm of what's happened in the field: ==A lot of time with no progress, and then in the last few years, a lot of progress seemingly out of nowhere.==

And now, in Spring 2023:
![[Pasted image 20240417182742.png]]
Fluent on all counts, from text-davinci-002 and text-davinci-003


Suggested readings:
- "On our best behavior" by Hector Levesque
	- We should come up with examples that test whether models deeply understand! *Inspired* by the [[Winograd]] Schema Challenge.
	- Technique: Impose very unlikely questions where humans have obvious answers
		- =="Could a Crocadile run the Steeple Chase?"==
			- You've probably never thought about this, but it's obvious
		- =="Are professional baseball players allowed to glue wings on their caps?"==

![[Pasted image 20240417182929.png]]
Uh oh, two confident answers that are contradictory, across closely-related models... interesting!









