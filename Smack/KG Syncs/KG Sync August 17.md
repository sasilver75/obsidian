
Did Navy Project Overmatch

NEC coming up

Ground Radar and SmackSesnsors
- We laod parametric data onto smack sensors; Type 12 radar...Nothing that qguarantees that those give us targetable TQs... 
	- We call into smack sensors, get an expected TLE out of it...
	- The issue is that  the entirety of the southwest islands defense is surface search radars and anti-search cruise missiles.


COA Graph
- ...





COA-Graph-Roadmap-DOI
- Something about capabilities... if I had to give it to a ihgih level overview... the idea is a persisted mission state... when we make a plan... that needs to map to a graph artifact (not necessarily a KG one, but something that... help us with decision process...)
- He's going to share a document about this
- Might not be in the KG at all, might be though. Going to be in partnership with the tech team going forwards. Not always mapping to what we need to do.
- We're co-opting the term multifidelity; talking about the graph an operational level; a single node on the graph is a n entire COA or something... so this nesting... we're taking it and using it.
- We're rapidly establishing.... bureaucratic artifact generation... but tryin got build some semblance of a process of everything we do at Smack. CJ has broken down "delivery cadence" into roughly quarterly buckets.... so by September... this thing has emerged as the front runner. Things that also fell in that bucket... might be LogNet...
- I won't be displeased if we have a v0 COA Graph in September...
	- SAM: CRADA, KG Resolvers, internal tooling, etc...


If the COA Graph represents plans, etc. at echelon (tactical, operational, strategic), CLEAR output is something that could be encode in a COA graph. Unless CLEAR allows for contingency planning (if this then this), ten it's only ever capable of encoding a single path (state one, state two, state three...) a full COA graph should have state 1 -> Stat 2, based on what happens in State 2, these are places we could goo...


Horizontal graph scaling: What's the next thing to upload?
- JC: It's likely to be formalizing INDOPACOM
- O: Ray wants me to jump in the rest of China

If SAILS goes south, I don't want "Neo4j is the problem" to be the story.
Which would obfuscate some more 


SOLIC relying heavily on the concept of persisted mission state...


We might be folding Node-Init into KGTS

Something that Eli flagged on this over the weekend... is for Shared properties... speed being one of them (effectors, launchers)... because the nomenclature is similar... it appeared to be the case that in some situations the platform was being assigned the munition speed... we want to be cognizant of it.

JC: The Node Initializer could be context-injected with a description of the platform.
N: Some of the descriptions... I suspect that some of the descriptions themselves are suffering from the same issue (e.g. launcher being confused for the munition).

JC Re: descriptions: As a traversal... Type 12 Launcher IS Ground Launcher IS... how do I quickly do a traversal and string packaging of its taxonomy... so that we can autofill descriptions based on its semantic taxonomy.

COA graph should be generatable from an OPORD... 
COA Graph is strongly decouple d from the whole decomposing the JPP thing that JC has been doing.

What creates that OPORD is still


When talking to 

SatOps: Do we have anything at Smack that uses structured satellite operations, the answer is no, so why are we talking to the people? Everything is chaotic and we're just figuring it, but #2 is what John Falcone calls the Shotgun strategy.

DEN stuff shifts... was escort, now it's looking at [[Air Interdiction of Maritime Target|AIMT]]... MVP that we show clients is fluid... I suspect that there's more work to do... for the GRaph to do, and more experts to egnage in than is possible, and so we'd want to hve a structured way to do that... because Olivia quickly will not be able to talk to everyone that John has.
- But we ant to record the meetings, because if you're not there to ask the right graph architecture questions, it won't matter.


