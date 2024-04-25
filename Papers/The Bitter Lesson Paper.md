March 13, 2019 -- [[Richard Sutton]]
Link: [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

A paper from Rich Sutton in which he notes that across any fields, researchers who have stubbornly tried to introduce bias into models, "engineering knowledge into them" have been defeated by Moore's law combined with general learning algorithms with small amounts of bias. We aren't good at engineering opinionated algorithms; we should let 

The claim
>   The biggest lesson that can be read from 70 years of AI research is that ==general methods that leverage computation are ultimately the most effective, and by a large margin==.

Re: Chess
> When a simpler, search-based approach with special hardware and software proved vastly more effective, these human-knowledge-based chess researchers were not good losers. They said that ``brute force" search may have won this time, but it was not a general strategy, and anyway it was not how people played chess. These ==researchers wanted methods based on human input to win and were disappointed when they did not==.

Re: Go
> Enormous initial efforts went into avoiding search by taking advantage of human knowledge, or of the special features of the game, but all those efforts proved irrelevant, or worse, once search was applied effectively at scale.
> In computer Go, as in computer chess, ==researchers' initial effort was directed towards utilizing human understanding== (so that less search was needed) and only much later was ==much greater success had by embracing search and learning==.

Re: Speech Recognition
> As in the games, ==researchers== always tried to make systems that worked the way the researchers thought their own minds worked---they ==tried to put that knowledge in their systems==---but it proved ultimately counterproductive, and a ==colossal waste of researcher's time==, when, through Moore's law, massive computation became available and a means was found to put it to good use.

The First General Lesson
> The bitter lesson is based on the historical observations that ==1) AI researchers have often tried to build knowledge into their agents, 2) this always helps in the short term, and is personally satisfying to the researcher, but 3) in the long run it plateaus and even inhibits further progress, and 4) breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning.== The eventual success is tinged with bitterness, and often incompletely digested, because it is success over a favored, human-centric approach.
> One thing that should be learned from the bitter lesson is ==the great power of general purpose methods==, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are ==search== and ==learning==.

The Second General Lesson
> The second general point to be learned from the bitter lesson is that ==the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds==, such as simple ways to think about space, objects, multiple agents, or symmetries. All these are part of the arbitrary, intrinsically-complex, outside world. They are not what should be built in, as their complexity is endless; ==instead we should build in only the meta-methods that can find and capture this arbitrary complexity==. Essential to these methods is that they can find good approximations, but the search for them should be by our methods, not by us. We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done.
> 









