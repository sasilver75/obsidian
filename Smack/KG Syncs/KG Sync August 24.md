
https://smack.sharepoint.us/sites/DecisionDominance/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FDecisionDominance%2FShared%20Documents%2FTeam%20Folders%2FBHG%2FWeekly%2F24%20Aug%20Sync%2Emd&parent=%2Fsites%2FDecisionDominance%2FShared%20Documents%2FTeam%20Folders%2FBHG%2FWeekly&p=true&ga=1


NEC Demo went great
- Will Schlaegle when asked if they will ask alot of technical questions, sad "No!" But they basically drilled John Coogan and toa lesser extent Ray for an hour. It's what we shud have expected from the Japanese.
	- They don't do the US defense contractor vibe of "That looks like a cool UI, we'll take you by your "
	- We filmed the NEC walkthrough and full south china sea scneario for Ray to go out to the ___ conference. They will be sitting in front of Paparo and other general officers... looking at tens of stars in the room.




PLANE: We're hosting our own version of PLANE that's a linear equivalent that's capable of sharing CUI (?) data:
![[Pasted image 20260824110516.png]]
Tonight, he's going to have our scaffolding of stuff, and turn our scaffolding over to project owners, and we'll move forwards.
You can get an access token to have Claude etc talk to Plane. Work smarter, not harder.

I want to be disciplined about hte Plane buildout.
If you aenaren't disciplined about whathte hierarchy of stuff is, you tend to end up with a massive backlog, where something shoulds be proejctsa nd other thing shouldn't be on there. If you aren't disciplined, it should have a strong tendenency towards chaos. Now we'ere entering a new ecosystem of new stuff. How we step off this week... is going to go a long way towards making our lives easier. Any of us could look at Plane, here's my Bastion, here's thing s that peoplew ant from me, etc. DJ and Sam for instance have been adfrift, and John wants us not to be there.

Going to let the COA Graph DOI sit


Olivia:
- Checking/Cleaning NTC, found the typical 



WeaponNetv2 with some help from Tater
- Not sure what the 
We're definitely going to be redoing SenseNet
There's some SemNet and CapeNet weird overlap, so there will be some weird refactor stuff in there
That will let us step off smartly on things like LogNet



	There are inescapable gravity wells that you can't escape. Waht youw ant to avoid... everyone is drowning and they want a flotation device, and they'll look at something you propose and turn it into a flotation device. Multiidelity RL is a thing to be cautious about. It has merits... but we don't want it showing up in a lot of diagrams, whichs ays "The RL pipeline will succeed only if RL succeeds." We don't want to createa kestone artifact about it as if it's already been done. I'm giving you advice teo be cognizant of it's hipocritical because I'm allowing the COA graph to grow legs and run. 


The distinction between where graph architecturea nd business logic lies...
People will sasy "Graph will do something that business logic will do"


Sam
- JC: A bod of work that would be great is the... tech model data pipelines.
	- For C2E...The C2E pipeline is brittle, kind of dogshit, and very confusing. We'd like to probably take time... and basically rip it out and do it like an adult. We're in a position where we can. John will want to talk with me, and RPC...
	- We want to ... have an extensible system/pattern by which we create those in a knowable way, which is hard. Lot of design decisions there. Decoupling the pipeline/model, and establishing the working framework... and talking to ELI and asking what his API requirements are, satisfying those.
	- All of those model currently pull from the graph. At least the first pass rip -- nobody's going to contest that space because no one wants to touch it with a six foot pole. It's reasonable for the graph team to own model-graph connectors. Some deployment things get app-engy (where to put this pipeline in our ecosystem, how should it be envoked in temporal, etc).
	- RL, RL/OR hybrid, whatever... the models we deploy today are the only examples of true Smack decide modes that take data nd make a decision. Lot of talk about the RL pipeline, we love the RL pipeline.  Our ability to effectively onboard new models is critical for many purposes. We have sold to VCs that we're going to use RL... we've always riffed things as a v0 that we're never going to go back, and that's what the current C2E pipeline is. We've never made it right. This causes reputational injury to the current OR solution we have, and we spend a lot of time debugging a very shiesty pipeline, which is the combination of 3 IC's amalgam of bullshit... lots of benefits to owning authoritatively, doing it right, and stepping off with an extensible pattern.
	- Almost nothing at SMACK is going to be deprioritized by the coherence dashboard, which is an internal tool. We're getting by with our understanding of the database.

SOCPAC CTO said that they won't advocate for a contract unless they have fingers on keyboard.

