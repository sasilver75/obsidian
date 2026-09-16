
These are some questions that I'd have had (I think) towards the person who was actually asking for us to build this, or questions that I would have for the other engineers.


Questions that I'd have, given the posed problem:
- Do BLUE units have a loadout (which implies specific capacity for specific RED platform effects)
- Do RED units have a loadout (which implies specific capacity for specific BLUE platform effects)
- Where will RED units start?
- Where will BLUE units start? Do they need to all be able to return to the *same* base after striking? Just the air units?
- How do we want to model maneuverability of BLUE units?
- How do we want to model RED threat to BLUE units?
- Do we treat RED units as static?
- Can a single unit perform both an ISR and a Strike role in the solve?
- Can ground, air, and maritime units all provide ISR? Of what sort of targets?
- What is ISR?
- What is TQ?
- How does ISR provide TQ?
- What is the use of TQ?
- What level of TQ is required during cue/launch/inflight/terminal/combat assessment for a target?
- In the context of the unified dashboard, does this produced plan have to dovetail with the generated COA in P2C?
- How should weaponeering work?
	- What sort of effects framework/desired effects are we going to consider?
- How should heterogeneous weaponeering work?
- Should we plan for reattack?
- Should we have any sort of actual FSCAs? If we have a ground launcher shooting from A->B, do we have a goalpost-shaped ROZ between them? Or generally along munition flight paths?
- What about abort/replan rules?



Questions that I have, given our existing implementation:
- Why did we choose to cap the threat contribution of an individual platform, but not cap the threat score summation from multiple RED platforms?
- To have the Generation Point information optionally coming from information in the knowledge graph... that seems like a weird mixing of concerns. Why should the knowledge graph have to care about generation points? If anything, there should be a relationship from the instances to their home base infrastructure/formation whatever it might be. Not a weird "generation point" foreign-key-via-name, right? I think I have something in my scruples about this.
- Is this just a cleanup, or is this a rewrite? I heard people say things like "We shouldn't care about Satellites", etc. Is the replaced product supposed to be one that ... still satisfies existing customers? Is it 
- The existing solver code also determines the layout (of all that are availbale, which is one per platform type afaict) for every scheduled blue unit... this makes sense maybe for aircraft that are starting in airbase generation points and shit, but don't we also have some units that are starting out in the field? Things like ground or even maritime assets? Does it make sense to magically be able to select the loadout of these various units at scheduling time? I don't think that Athena laydowns include SELECTED loadouts, do they?
	- No, they don't; It seems like the actual ElementTypes are the things that (sometimes) have LoadoutTypes associated with them. And these ElementTypes aren't copied over in an AthenaScenario... just 



Questions that I have of the project:
- So the idea is that this is a drop-in replacement for the work that we have. That mostly speaks to external (i.e. product-facing) interfaces, to me. Is the assumption even that Eli won't be doing any (re) modeling for the project either? If we were to determine that we want to do something different (adding additional information, removing certain constraints, whatever it might be), it would probably require a different model.
- There's a weird thing happening in the code that only enemy units in TAIs contribute in a "targetable" (non-durable) way to the TWM. A principal or residual target outside a TWM still contributes to the TWM even when destroyed.



Bugs I'm seeing:
- ...



