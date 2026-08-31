https://smack.sharepoint.us/sites/DecisionDominance/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FDecisionDominance%2FShared%20Documents%2FTeam%20Folders%2FBHG%2FWeekly%2F31%20Aug%20Sync%2Emd&parent=%2Fsites%2FDecisionDominance%2FShared%20Documents%2FTeam%20Folders%2FBHG%2FWeekly&p=true&ga=1

PLANE buildout continuing...
- Model Process ROC: Rehearsal of Contracts (Sam w/ the C2E Process Exercise)
- KG Exposure Kickoff (DJ with SAGE-Gen -> Dany+Steve engagement)
	- They have done a spectacular scope creep; JC is reeling from that. Now there are four things from SOLIC, not there; in addition to Alpha, Omega MSS, Omega Mars, also an Omega MARFORPAC integration, and they want to extend it into multiple warfighting functions.
	- Natural language is not good enough; we know it's not good enough, so we want to be identifying places where we can AT LEAST get graph traversal; the lowest hanging fruit of how to get SAGE workflows, which are just table stakes agents... and get it so that it can cite back to things that Smack owns th others don't have in, in lieu of real models (Sam model pipleline), which is high latency. The low hanging fruit is getting a graph traversal in there to cite it.
	- One is us after the call going over SAGE skills and how it looks. The second one is grabbing Dany/Steve and telling us how IW works.
	- Scope creepgoes beyond what our team... can play defense on. We'll get thre.
- CLEAR support (enduring)
- Horizontal graph scaling (Enduring), Rest of China
- Coherence dashboard; No one cares, but JC cares still. 

Tomorrow there will be an internal demo that JC will be giving to Growth people.
==Sam== can get an invite to this

For the Epics... I think that I'm in "Feature Iteration"

A lot of our epics end Sep 30
CJ is attempting to establish a delivery cadence... do a quarterly thing. So we want to look where we want to be by Sep 30.... in a reasonable, achievable way, and backplan some tickets off of it. 
- Any graph arch stuff, JC is considering Nick/Olivia a single team on that. N/O will architect on where we wanna be.
- For DJ and sam, we'll be talkign a fair amount this week, but that Sep30 should probably.... you want to focus on C2E pipeline (Sam) and IW buildout (DJ)... and try to get one step that can be backed by a traversal.
- The end... success state is that we have started using PLANE to backplan off of objectives, and that we have a good feeling about.


JC would like... to totally mitigate/minimize our need to understand anything about PLANE. Theres hould be no overhead other than adding tickets and chaining them together. The ticket where I want to be... 

KG vs Business Logic
- In escort protection there's something called the PEPZ: Proetecte entity protected zone or something.. which which is defined by thte therats in the battlespace. In that way, the fact that an F-35 CAN escort a bomber... the radius of a PEPZ is something that needs t obe computed at runtime; this is not something that could be resolved in the graph.
- Loose definition: Is it static or state-agnostic, vs. informed by state
	- This is sort of informed by state; if I wanted to traverse a graph to understand what munitions I could use to prune options. As you interface with sim team... they're going to freqently... not have a great understanding of wherre business logic ilke that residse. The fundamental core of that is physics-based modeling. If I need to know the range of a radar... I'm giong to compute the radar range equation. The parameters of that funtion canll can reside in the KG, but determinizing the range of ar adar at runtime is not a KG thing. There's confusion there. If people want business logic things... we have to call it out
		- We can call services to execute that logic independently.
		- It's likely that SIM and CLEAR business logic are reused, so we're probably expecting a tertiary service that holds our physics-based models. 
		- the current example for that is smack-sensors, but we'd want that for all things,.
		- The takeaway is: That's not the brain housing group's job. We can help, inform, do what we can, but the KG does not own physics based modeling, nor should it. 
	- Often times yo'll find (usually as a legacy of leo) that the KG can do things that it cant do.


It's nice to retroactively add to PLANE the process you did.