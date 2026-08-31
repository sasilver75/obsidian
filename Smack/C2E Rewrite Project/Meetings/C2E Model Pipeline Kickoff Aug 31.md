# Agenda

> May not need the whole hour. Looking to hit the first stage in the process and get our feet under us. Objectives:
1. List the 'requirements' of C2E as we understand them today (or more specifically as we wish they were articulated when we began)
2. Work to define what that artifact should look like. MD template for Plane Pages

__________
## Pre-Meeting Work:

Central requirement:
- Given an approved operational concept, selected targets, available forces, and planning constraints, C2E must produce a feasible, explainable strike-and-ISR plan that an operator can review, revise, approve, and hand off for execution.

Requirements:
- C2E must accept a defined planning scope, containing:
	- Approved ConOp and planning window,
	- RED targets under consideration, including identity, type, location, priority, and required platform/effects (if any?).
		- What determines red targets?
	- Available BLUE asset instances and their current locations
		- What determines available blue assets?
	- Platform, weapon, sensor, capacity for RED/BLUE targets
	- Relevant control measures and geographic restrictions
	- Planning policies such as P_kill thresholds, required track quality, acceptable risk to units, return-to-base rules
	- The system should freeze or version this inputs as a planning snapshot; a later change to the knowledge graph must not silently change the meaning of an existing run, right?

Questions to ask:
- What problem is C2E responsible for solving?
	- What is the primary user of C2E, and what decision is that person trying to make?
	- What does C2E receive from the upstream planning process?
	- Where does C2E's responsibility end?
	- Is there any work that belongs outside C2E even if the current implementation happens to perform it?
- How are we formalizing/simplifying the problem?
- What is the planning input?
	- What must an approved planning package contain before C2E can start?
	- Which input fields are required, optional, or derived?
	- What's the authoritative source for targets, assets, weapons, sensors, geometry, and threat data?
	- Is the input frozen when planning begins, or can live data changes affect an active run?
	- How are unavailable, stale, contradictory, or incomplete inputs handled?
	- What version identifiers must be recorded so that a result can be reproduced?
	- Can we describe one immutable object that completely represents the planning problem submitted to C2E?
- What do operator selections mean?
	- Does selecting a target mean that it MUST appear in the plan?
	- Does target rank affect feasibility, objective importance, execution order, or display order?
	- Does selecting a platform for a target mandate it, or prefer it? 
		- What if it can't be used (either because platform/munition can't get in range, or is assigned to higher target, can C2E substitute)?
		- What logic dictates what platforms are selectable in the dropdown UI?
	- Does selecting a munition for a target mandate it, or prefer it? What if it can't be used, can C2E substitute?
		- What if it can't be used (either because platform/munition can't get in range, or is assigned to higher target, can C2E substitute)?
		- What logic dictates what platforms are selectable in the dropdown UI?
	- Can C2E add supporting assets that the operator didn't select?
	- What do we do if a selected combination of effector/platform/target is infeasible, should C2E fail/suggest alternatives/silently substitute something else?
- How is resource availability defined?
	- We use specific asset instances, yes?
	- What makes an asset "available" for this operation?
	- Are maintenance status, fuel, location, loadout, readiness, and existing taskings considered? What about future availbility and resupply? Do we allow for planes to do integrated combat turns, etc?
	- Is an asset reserved when a plan is generated, approved, or published? e.g. if you're creating multiple overlapping operations.
- What is a feasible strike?
	- What conditions must be satisfied before a target is considered serviceable?
	- Does a platform need to be able to return home?
	- What launch/release locations are allowed?
	- How are platform range, weapon range, fuel, speed, and route duration applied?
	- Is a minimum probability of effect required?
	- Can one platform service multiple targets?
	- Are targets struck once, or can the plan schedule repeated effects? e.g. do we have to have slack/budget for some expected amount of reattack?
	- How do we determine time on target, given a release time of a munition from a platform?
	- Are RED units that are not specified as targets actually untargeable? Even if it's a SAM launcher sitting at the end of our runway that makes our plans look terrible, but didn't make the target list for the operation?
- ISR Support
	- What targets require ISR support?
	- What target/track-quality level is required?
	- At what moments is that quality required?  Does this depend on the effector type?
		- Cueing
		- Launch
		- In-Flight update
		- Terminal engagement
		- Assessment
	- Do we need continuous custody, or are gaps permitted? Is it just that we need to maintain a certain TQ... at certain times?
	- Can multiple sensors combine to satisfy one requirement? What's the effect on TQ of having two things look at a thing, versus one thing? Does TQ "jump," or do you need time on station to raise TQ before it then degrades?
	- What makes a ISR platform/sensor/target combination eligible?
	- What should happen if required target quality can't be achieved? Does that make a strike infeasible?
- How should threat and route risk be tolerated?
	- How do we calculate "threat" or "risk"?
		- Does it depend on direction, altitude, time, or platform configuration? Does it depend on sensor coverage? RCS? etc.
		- Does it depend on exposure time? Is it fine to exposure yourself to high risk for a small amount of time? What level of exposure can an operator accept? Does this depend on whether it's a manned platform?
	- What locations are valid travel nodes for each operational domain?
	- Is there a route risk above which travel is prohibited, versus just undesirable/adding cost.****
	- How should multiple threats combine? Do we add their costs? Take the max?
	- Do we treat our threat calculation as if we know where all enemy units are? Is there any sort of "uncertainty" about a long route through enemy airspace that doesn't seem to have any red units.
- What is C2E optimizing?
	- Do we want to service everyt selected target, or as many targets as possible?
	- How do target priority, probability of effect, route distance, threat exposure, weapon expenditure, and asset utilization trade off?
		- Are these somehow configurable?
	- Should C2E produce one answer, or multiple alternatives? What could make one plan better than another? 
	- Should repeated runs with identical inputs produce the same result?
- What should the result contain?
	- Which exact tasking fields are required for each assigned aset?
	- Must the result include routes, times, release points, weapon quantities, ISR windows, and return plans?
	- How are unassigned asset and unserviced targets represented.
	- What warnings/incomplete-data conditions must be visible, if at all?
	- ==What does the downstream execution product require?==
	- Is the result a complete replacement of downstream state, or a proposed change?
- How should failure/infeasability be communicated?
	- What's the difference between invalid input, incomplete data, no feasible plan, solver failure, or infrastructure failure?
	- What explanation must accompany an unserviceable target?
	- Can a partially successful plan be returned?
	- Which failures block approval?
	- "No solution" shouldn't be the complete product response.
- How does the operator review and modify a plan?
	- Can the operator lock individual assignments and ask C2E to replan the remainder?
	- Can constraints or preferences be changed after generation?
	- Are prior results preserved?
	- Can approved plans be superseded, withdrawn, or amended?
- What is the lifecycle of a planning run?
	- What event creates a run?
	- What event starts computation?
	- What are valid states of the lifecycle?
	- What dose cancellation guarantee?
	- What happens if a stage or service restarts?

Important questions:
1. Are selected platforms and munition values requirements or preferences?
2. Who chooses weapon quantity?
3. Must every approved target appear in the result?
4. What is the authoritative definition of an available asset? May C2E add ISR or strike assets without operator selection?
5. What makes a movement route valid?
6. What does missing threat information mean?
7. What makes ISR mandatory for a strike? What is the requirement of ISR?
8. What must be preserved when a plan is rerun or published?
9. What constitutes a useful no-solution explanation?
10. What exact planning snapshot must be retained to reproduce the result?
11. (Clarifying): What information from the P2C stage is actually used? We approve a Concept of Fires in step 5, and then in 6 we get some FSTs generated, and we have then priorities across these FSTs that we can set? Is it those?


Something interesting that we do is creating the TWM before doing the sensor stuff in ETL (which at least contains the satellite sensors, etc). I feel like stepping back and understanding what sort of data would be needed for a threat-weighted map. (And before that, understanding why we need a threat-weighted map, or why that needs to be over a graphical route system, either automatically derived or determined by the user input).

Satellites during precompute: During precompute, C2E obtains predicted satellite observation opportunities; each opportunity identifies {the satellite, the target being observed, the predicted observation time, the sensor modality, the predicted TLE}.
- I wonder how realistic what we're doing here 

TQ is how targeteers think about the world. We map it directly to TLE, but there are other people that... map it directly to time. Some other people call it "Time Quality," not "Track Quality," and it's "how old is this information?" Tehy take TLE and multiply the speed and time latency to expand that... We don't do that. Mainly because we're doing predictive planning.


_______

# Meeting

If we were to enumerate the requirements, they would be...


Disposition meaning: Laydown: Force composition, and where they are in the world.




Sam Questions:
- When we're given a friendly disposition, 
- What is a weapon to target pairing? Does it include a time? Launch point? Navigation?
- What is a "Smart munition" that requires these updates?
	- JC says that only few of our munitions are actually smart munitions
- When we say that munitions have an in flight TQ, during what portion of the flight is a certain TQ required?
	- It seems like we had made an assumption last time that the in-flight TQ gate was like midway through
	- But that Eli sometimes turns off TQ gates for midflight... depending on the demo, or maybe even for launch too? Not sure.
- What is a Cue
- Sam followup: Do we support heterogenous salvos.

Eli:
- Depending on context of demo, I've been overriding TQ arbitrarily.... I make sure everything has a cue, but if I see a certain demo happening, I might say "no in-flights," or maybe add in-flights if we want to see more action on the.


We associate a munition to one of several TQ gates.. we map a TLE to a TQ at runtime.


Re: TLE -> TQ: TQ of 15 is essentially 0 TLE, and TQ of 1 is hundreds of kilometers


JC: You should launch on a sufficient TQ, and if you can make it better, you can, but you shouldn't bank on an in-flight track update.


JMEMS: There was some homogenous salvo size that corresponding to a k-kill. "This is the number of things you need to shoot at a thing". We refer to this as... the median effect. If I was to consult the JMEMS manual... and say "Type 55, LRASM," it will give me a number of LRASM. That's not including networked effects like escort platforms that might intercept it, not including time of day, obscurance, weather, all those things.

TODO: Eli find somes document

If we were to look at the shit we were served on a platter here... and were to say... I want this to be well-specified in a template... what's the verbiage we would use?
- We want to get from: 
	- Someone with expertise has an idea
	- To very easily identifiable modeling constraints.
- If you're trying to shunt it into the recursive thing we had, that thing isn't expressive enough to do full baseline boilerplate new thing all the way to model, it's more for smaller iterations. Not my circus not my monkeys.

There's likely a number of associated paradigms that translate to modeling constraints, and we want to tease that out of the subject matter expert at the outset.





