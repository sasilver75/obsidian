#article  #premium 
Link: https://www.noemamag.com/ai-and-the-limits-of-language/
[[Yann LeCun]]

Insightful Excerpts:

> This notion (that knowledge is linguistic; that knowing something means "thinking the correct sentence") is also ==what underlies the Turing test== -- if ==the machine *says* everything that it's supposed to say, then that means it *knows* what it's talking about, since knowing the right sentences, and when to deploy them, surely *exhausts* knowledge==.
> - This has been subject to a withering critique since it was introduced -- ==language doesn't exhaust knowledge==; on the contrary, it's only a highly-specific, deeply limited kind of knowledge representation.
> - All language (programming, symbolic, spoken) turns on a specific type of representational schema -- ==it excels at expressing discrete objects and properties and the relationships between them at an extremely high level of abstraction==. But there is a massive difference between reading a musical score and listening to a recording of the music, and a further difference from having the skill to play it.

> ==All representational schemas involve a compression of information about something -- what gets left in and left out of the compression varies!==
> - The representational schema of ==language== ==struggles with more concrete information, such as describing irregular shapes, the specific motion of objects, the functioning of a complex mechanism, or the nuanced brushwork of a painting -- and much less the finicky, context-specific movements needed to surf a wave==.

> Language is important because it can convey a lot of information in a small format, and can involve reproducing and making it available widely; but compressing information in language isn't cost-free; it takes a LOT of effort to *decode* a dense passage!
> - A lot of time in Humanities classes is still spent going over difficult passages.

> Though they will undoubtedly *seem* to approximate it if we stick to the surface. And in many cases, ==the surface is enough! Most talk is small talk.==

> It is the deep, *nonlinguistic* understanding that humans have which is the grounding that makes language useful!

----
When a Google engineer recently declared Google's AI chatbot a person, pandemonium ensued! [[LaMDA]] predicted the next token so impressively that the engineer began believe that there was a ghost in the machine.

Perhaps deceiving humans isn't very challenging -- we see saints in toast, after all.

The diversity of responses to this news story highlights a problem: As LLMs become more common and powerful, there seems to be less and less agreement over how we should understand them!

they have many capabilities, but often fall on their face when it comes to "common sense" tasks -- how can these systems be so smart, yet also so limited?

The underlying problem isn't the AI -- it's the problem of the limited nature of *language* -- once we abandon old assumptions about the connection between thought and language, it's clear that these systems are doomed to a shallow understanding that will never reach the thinking we see in humans.

# Saying it all
- A dominant theme for much for much of the 19th and 20th century in philosophy and science is that ==knowledge *just is* linguistic -- that knowing something means *thinking the correct sentence*==, and grasping how it connects to others sentences in a big web of all the claims we know.
- The ideal form of language, in this logic, would be some purely formal, logical-mathematical one composed of arbitrary symbols connected by strict rules of inference.
	- Wittgeinstein: "The totality of true propositions is the whole of natural science."

- This view is still assumed by some overeducated, intellectual types ((Shots fired at Chomsky)) -- that everything that can be known can be contained in an encyclopedia, therefore reading everything might give us comprehensive knowledge of everything.
	- It motivated a lot of the early work in Symbolic AI, where symbol manipulation (arbitrary symbols bound together in different ways, according to logical rules) was the default paradigm.
	- This notion (that knowledge is linguistic; that knowing something means "thinking the correct sentence") is also ==what underlies the Turing test== -- if ==the machine *says* everything that it's supposed to say, then that means it *knows* what it's talking about, since knowing the right sentences, and when to deploy them, surely *exhausts* knowledge==.
		- This has been subject to a withering critique since it was introduced -- ==language doesn't exhaust knowledge==; on the contrary, it's only a highly-specific, deeply limited kind of knowledge representation.
		- All language (programming, symbolic, spoken) turns on a specific type of representational schema -- ==it excels at expressing discrete objects and properties and the relationships between them at an extremely high level of abstraction==. But there is a massive difference between reading a musical score and listening to a recording of the music, and a further difference from having the skill to play it.

==All representational schemas involve a compression of information about something -- what gets left in and left out of the compression varies!==
- The representational schema of ==language== ==struggles with more concrete information, such as describing irregular shapes, the specific motion of objects, the functioning of a complex mechanism, or the nuanced brushwork of a painting -- and much less the finicky, context-specific movements needed to surf a wave==.

# The Limits of Language
- One way of grasping what is distinctive about the linguistic representational schema -- and how it's limited -- is recognizing how little information it passes along on its own.
- Language is a very low-bandwidth method for transmitting information -- isolated words or sentences convey very little, and many sentences are deeply ambiguous and context-dependent.
- Humans don't need the perfect vehicle for communication, because we share a nonlinguistic understanding -- our understanding of a sentence often depends on our deeper understanding of the contexts in which this kind of sentence shows up, allowing us to infer what it is trying to say.
	- Research suggests the amount of background knowledge a child has on a topic is actually the key factor of comprehension.

- The inherently contextual nature of words and sentences is at the heart of how LLMs work -- In LLMs, the system discerns patterns at multiple levels in existing texts, seeing both how individual words are connected in the passage, but also how the sentences all hang together within the larger passage which frames them.
- ==The result is that its grasp of language is ineliminably contextual; every word is understood not on its dictionary meaning, but in terms of the role it plays in a diverse collection of sentences.==
- In short, LLMS are trained to pick up on the background knowledge for each sentence, looking to the surrounding words and sentences to piece together what's going on.

# Shallow understanding
- Some balk at using the term "understanding" in this context, or calling LLMs "intelligent"; it isn't clear what semantic gatekeeping is buying anyone these days... but critics are right to accuse these systems of being engaged in a kind of mimicry.
- LLMs have acquired this kind of shallow understanding (like students parroting jargon from their professors) about everything. A system like GPT-3 is trained by masking future words in a sentence and forcing the machine to guess what word is most likely, then being corrected for bad guesses.
- The ability to explain a concept linguistically is different from the ability to *use* it practically:
	- The system can explain how to perform long division without being able to perform it itself.
	- The system will explain what words are offensive, and then blithely go on to say them.

The latter kind of know-how is essential to language users, but it doesn't make them linguistic skills -- the linguistic component is incidental, not the main thing.

((LeCun then goes on a minirant about how these systems have memories of only a few paragraphs (wait a year), and are bad at active listening, recall, and revisiting prior comments, sticking to a topic to make a specific point while fending off distractors, and so on.))

Language is important because it can convey a lot of information in a small format, and can involve reproducing and making it available widely; but compressing information in language isn't cost-free; it takes a LOT of effort to *decode* a dense passage!
- A lot of time in Humanities classes is still spent going over difficult passages.

This explains why a machine trained on human knowledge can know so much and yet so little; it's acquiring a small part of human knowledge through a tiny bottleneck... but that small part of human knowledge can be about *anything*, whether it's love or astrophysics.

# Exorcising the Ghost
- This doesn't make our machines stupid, but it ==suggests limits to how smart these systems can be.==

> "A system trained on language alone will never approximate human intelligence, even if trained from now until the heat death of the universe."

- Though they will undoubtedly *seem* to approximate it if we stick to the surface. And in many cases, ==the surface is enough! Most talk is small talk.==
	- But we shouldn't confuse the shallow understanding LLMs possess for the deep understanding humans acquire from watching the spectacle of the world, exploring it, and experimenting in interacting with culture and other people.

==It is the deep, *nonlingnuistic* understanding that humans have which is the grounding that makes language useful==! It's because we possess a deep understanding of the world that we can quickly understand what other people are talking about.

This broader, context-sensitive kind of learning is the most basic and ancient kind of knowledge, and underlies the emergence of sentience in embodied critters, and makes it possible to survive and flourish. It's what AI researchers focus on when searching for common sense in AI.
- LLMs have no stable body or abiding world to be sentient *of* -- so their knowledge begins and ends with more words and their common sense is always skin deep.
- ((This seems to be an argument for the embodiment hypothesis))

gg






