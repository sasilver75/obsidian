#article 
Link: https://medium.com/@colin.fraser/generative-ai-is-a-hammer-and-no-one-knows-what-is-and-isnt-a-nail-4c7f3f0911aa

----

Imagine a world without hammers
- Driving nails into the wall with your bare fists
- Kicking through drywall with your foot to take down a wall
- Tying your text stake to a rock to stop it from flying away in the wind

Imagine this world has a long history of scientific research, as well as a tradition of speculative science fiction and futurism centered around the concept of "Artificial Labor" (AL).

One day, it's anticipated that AL will drive your nails, tear down your walls, and stake your tent 😱⛺
In the far future, Artificial General Labor (AGL) will wash the dishes, do the laundry, walk the dog, pickup the kids from school, and essentially do every annoying laborious task you can imagine, freeing you from a life of drudgery.

A company called OpenAL, then, ~"invents" the hammer.
- We can suddenly drive nails much faster and more effectively than ever before.
- Thousands of AL-first technology firms materialize out of nowhere to do things like stake down tents using the latest AL tools from OpenAL.
- Independent hackers figure out how to build their own hammers from materials lying around the house, publishing their instructions online for free.

For many people, it seems like we're on our way to a WALL-E AGL scenario -- OpenAL even says that it's their whole purpose to bring this AGL about!

But wait a second -- in the WALL-E scenario, AL is doing the dishes... but hammers can't do dishes, right?
- Well... Hammer4 looks pretty good... some people are saying that maybe Hammer5 will be able to do it! Because Hammer5 will cost many trillions of dollars to construct.


If you're not getting it yet, he's suggesting that ==ChatGPT was like the release of the hammer==; huge generative models like ChatGPT, Stable Diffusion Sora, etc. are a new and surprising subcategory of AI technology with a wide and rapidly-expanding range of potential uses.

==But there are some things that ChatGPT seems to be quite bad at, which we wouldn't think were very difficult for either humans or traditional computers programs== (he shows the "sum to 22" game, where you take turns picking numbers 1-7, and whoever gets the number to 22 wins).

The most common point of view about the trajectory of AI among people who are into AI is that "*if there is something like the sum-to-22 game that the AI can't do today, surely it will be able to do it very soon! It's just a matter of time!*"
- The author thinks that the above quote is not an accurate picture of the current state of things


#### There's no one thing called "AI"
- AI is a huge and fuzzy category comprising everything from chess engines to search engines to facial recognition to robot dogs to the operating system from the movie *Her*.
- Technologies in the AI category all have different things that they *can* and *cannot* do.

![[Pasted image 20240326235736.png|300]]

AI is too broad and fuzzy to cleanly decompose into a proper hierarchy, but there are a few ways to impose a messy order on it.

At the broadest level, ==there's maybe a distinction between Symbolic AI and Machine Learning==

Under ML, you might have some subcategories like Classifiers or Recommenders, and one of these subcategories might be Generative AI. One of the categories below *this* might be LLM-based generation systems, of which ChatGPT is one example.

The point the author is trying to make is that ==ChatGPT is just one little point in a vast universe of technologies, somewhat analogously to how a hammer is one example from the general class of *tools*==

Frequently, reporting/media will collapse this huge category into a single amorphous entity.

A ==casual observer might reasonably surmise from these headlines that scientists at DeepMind are in possession of something called "an AI"== that is capable of doing all these various things.
And perhaps this AI at DeepMind is fundamentally the same sort of entity as ==ChatGPT, which also introduces itself as "an AI."==

All of this makes it seem like "an AI" is a discrete kind of thing that's manning chat bots, solving unsolved math problems, and beating high schoolers at geometry Olympiads -- but that isn't remotely the case!
- FunSearch, AlphaGeometry, and ChatGPT are three completely different kind of technologies which do three completely different kinds of things that are not at all interchangeable or even interoperable!

Something all three of these technologies have in common is that they are built using LLMs, and more generally that they are applications of this explosive new paradigm called Generative AI.
- This may make it seem like they are more connected than they actually are -- but they're extremely different applications of LLM.

There are many different things that count as "AI", and they all have very different properties.
- By sloppily mixing them, a picture is painted of a kind of system that doesn't actually exist, with an array of capabilities that no one thing has.

### A universal text generator is a universal hammer
- I can feel some people reading this post screaming that ChatGPT and a hammer is a category error -- hammers do just one thing -- hit things! ChatGPT on the other hand, generates text, *the universal interface* to a large variety of problems -- it's a *general* intelligence, right?
- But lurking below the surface of this is a very strong assumption -- ==the assumption that ChatGPT can generate *any* kind of text== -- and that all of the text necessary to perform any task can be generated by the specific procedure that ChatGPT uses to generate text.
- ==Strictly speaking, it's trivially false -- There's no way that ChatGPT can output the first billion decimal digits of $\pi$ , for example -- that's just not the type of task to which *its particular approach to generating text is suited*, but it's possible that this is the needed answer to a problem!==
	- My point is that there obviously exists at least one text generation task -- namely, this one -- that a system like ChatGPT cannot be expected to do -- ==there are *non-nails* to ChatGPT's hammer==!

The dominant view still is that "scale is all you need" -- that all it takes to build something that's good at that task is more computing power.
- In other words, this party thinks: =="If today's hammer can't do the dishes, all we need is a larger one!"==
	- There is at least one task -- outputting the decimal digits of $\pi$ that this kind of system can't do even in theory.

==It's clear that *artificial labor* can do many of these problematic tasks -- the question is whether a *hammer* can!==


#### A rough theory about which tasks are not nails
- ChatGPT works by making repeated guesses -- at any given point in its attempt to generate the right decimal digits of $\pi$ , there are 10 digits to choose from, only one of which is the right one -- ==the probability that it's going to make a million correct (even well-educated) guesses in a row is infinitesimally small -- so small that we might call it zero.==
- You can think of every individual word that ChatGPT generates as a little bet -- to generate its output, ChatGPT makes a sequence of discrete bets about the right token to select next.
	- If it happens to generate an incorrect word at any point, which it probably will, it can recover later -- no single suboptimal word will ruin the essay. ==For tasks where *betting correctly most of the time* can satisfy the criteria most of the time, ChatGPT's going to be okay most of the time.==
	- This contrasts sharply with the problem of printing digits of $\pi$ or playing the sum-to-22 game optimally -- in those cases, a single incorrect bet damns the whole output!


We see this same failure pattern in other Generative AI systems as well! There are a lot of ways to generate an image that looks like a bunch of elephants hanging out at the beach -- only a tiny fraction of those contain exactly seven elephants, so generating seven elephants is something that a GenAI system is going to have a hard time doing!

![[Pasted image 20240327003947.png|300]]

This doesn't improve much with scale -- the elephants will look better, but it will still generate some weird number of them.



This can be seen in in Sora as well

The prompt:
> "> A grandmother with neatly combed grey hair stands behind a colorful birthday cake with numerous candles at a wood dining room table, expression is one of pure joy and happiness, with a happy glow in her eye. She leans forward and blows out the candles with a gentle puff, the cake has pink frosting and sprinkles and the candles cease to flicker, the grandmother wears a light blue blouse adorned with floral patterns, several happy friends and family sitting at the table can be seen celebrating, out of focus. The scene is beautifully captured, cinematic, showing a 3/4 view of the grandmother and the dining room. Warm color tones and soft lighting enhance the mood"

![[Pasted image 20240327004227.png|300]]
In the video:
- People are not seated around the table
- It's not in 3/4 view
- She doesn't blow out the candles in the video

Artifacts:
- One candle has two flames
- Candle flames all pointing in different directions
- People in background are doing weird things in the video

==It seems that the generative AI strategy is good (and getting better) at *generating outputs that look similar to examples in training data*, but are NOT as good at generating output that satisfies specific criteria.==

The author doesn't believe that the basic generative AI strategy, which represents the problem of generating media as a random guessing game, is inherently well-suited to this particular task -- he doesn't think that we'll see the mass adoption that a lot of boosters expect, but realizes that he might look like an idiot in a year.

Coming back to text...

But wait, are some of ChatGPT's limitations overcome by the fact that it can write and run arbitrary computer programs (eg Python scripts)?
- This is just the magical universal hammer theory in disguise: The set of possible computer programs is big, and for ChatGPT to solve any arbitrary problem with a computer program, it would have to be able to write *any* computer programs!
	- {{But it does make SOME of the previously non-nail problems into nails, yes? Like the Pi one? I don't care for a world where there are no non-nail problems -- I just need a word where the vast majority of problems are hammerable! At what point would the author be on board, if not at 100%?}}
	- It seems, like text, that ==there are some computer programs that it's good at writing, and some that it's bad at writing, and the thing that separates these is something like the level of specificity required to satisfy the requirements.==

The authors hows that it's no better at generating a computer program (script) to play the sum-to-22 game than it is at playing the game itself!
Interestingly, the author showed ChatGPT as being unable to give the first 500 digits of Pi, both with and without access to its own interpreter.
![[Pasted image 20240327005606.png|400]]
Above: eh...

### No one knows which things are nails
- What can this technology do?
	- It can't generate digits of $\pi$
	- It can't generate an image of seven elephants
	- It can't generate a video of a grandmother blowing out birthday candles

==What are the nails? What can it do? How can you make money with it?==
- The author contends that he doesnt think anyone knows, in general.
- And that it doesn't seem like it's enough nails to warrant a seven-trillion-dollar investment -- we're going to need the hammer to be a lot more universal than that to make the economics work out.

![[Pasted image 20240327005849.png]]


==This whole issue is sidestepped if you get convinced by OpenAI that generative AI is a universal problem solver!==
If ChatGPT can do *everything*, then obviously ChatGPT can do your specific thing! ==If ChatGPT is *actually* a universal hammer, you don't even need to check if your problem is a nail!==
- ==For this reason, OpenAI and the rest of the ecosystem (chip manufacturers, AI-oriented VCs, cloud providers, newsletter writers, and OpenAI wrapper startups) all have a *very strong* incentive to embrace and spread the universal hammer theory!==  If you have a universal hammer, everyone is your customer.

Customer Service Chatbots feel closer to "recite the digits of $\pi$" side of the task spectrum, to the author.
- You want your bot to behave in a particular way, following a script and directing the customer to the right place at the right time, even when the customer behaves in unexpected ways. 
- You want it to behave in the way that a competent human agent would, in all situations -- but *NO ONE KNOWS HOW TO DO THIS!*

![[Pasted image 20240327010744.png]]

The random guessing nature of these things virtually guarantees that it will (at some point) output some nonsense (the so-called hallucination problem), and without knowing exactly how often this will happen and what kind of nonsense it will be, it's going to be very hard to use these in production in the way that is currently being promised.

==We're going to a see a huge wave of failed OpenAI API wrapper companies founded on the axiomatic belief that end-to-end Generative AI is the solution to every problem -- that it is a universal hammer.==

Chatbots as they currently exist aren't necessarily the best way to use the underlying technology!
- An LLM is a way to generate a certain type of text -- it's just something OpenAI tried for a laugh and people ended up getting really excited about it.

==The Hybrid approach to building with generative AI (eg AlphaGeometry, FunSearch) are completely different ways of using LLMs that have nothing to do with "chatting", but use the information contained within them *along with a deterministic decision-making module* to do generally-interesting and useful things!==


--------
--------
{Sam note from later. This is a preview (cherrypicked) pic from SD3; does scaling not "answer" this guy's problems with the granny, to some extent?}
![[Pasted image 20240328173939.png]]