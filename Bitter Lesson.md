A March 13, 2019 blog post from [[Richard Sutton]]
Blog: [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

A blog post from Rich Sutton about how leveraging compute and general learning techniques, rather than clever researcher tricks, is what works. Uses examples from Chess, Go, Speech Recognition, and Computer Vision.
He says it's ultimately a waste to try to cleverly engineer knowledge/priors into systems. Instead, we should build-in the meta-methods that can find and capture arbitrary complexity -- search and learning.

Excerpts
> "The biggest lesson that can be read from 70 years of AI research is that ==general methods that leverage computation are ultimately the most effective==, and by a large margin."

> "Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but ==the only thing that matters in the long run is the leveraging of computation==."

> This is a big lesson. As a field, we still have not thoroughly learned it, as we are continuing to make the same kind of mistakes. To see this, and to effectively resist it, we have to understand the appeal of these mistakes. We have to learn the bitter lesson that building in how we think we think does not work in the long run. The bitter lesson is based on the historical observations that ==1) AI researchers have often tried to build knowledge into their agents, 2) this always helps in the short term, and is personally satisfying to the researcher, but 3) in the long run it plateaus and even inhibits further progress, and 4) breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning.== The eventual success is tinged with bitterness, and often incompletely digested, because it is success over a favored, human-centric approach.

> One thing that should be learned from the bitter lesson is the ==great power of general purpose methods==, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are ==*search*== and ==*learning*==.

> The second general point to be learned from the bitter lesson is that ==the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries==. All these are part of the arbitrary, intrinsically-complex, outside world. ==They are not what should be built in, as their complexity is endless; instead we should build in only the meta-methods that can find and capture this arbitrary complexity==. Essential to these methods is that they can find good approximations, but the search for them should be by our methods, not by us. *We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done.*

![[Pasted image 20240620152821.png]]