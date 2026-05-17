Founded July 1, 2024, Headcount ~45 as of May 2026
Working on Tactical Resupply, building everything on top of Palantir Foundry
Emerged from Stealth in April 2025 after raised $15M from [[8VC]] and others (Silent Ventures, Timeless, Moonshots capital, Banter)


### Vocabulary
- "Cube": Means cargo volume (of a load).
- "Cube-out": A vehicle runs out of physical.
- "Weight-out": The vehicle hits its max payload weight before its cargo space is fuel.
- 

# Founding Team
- Woody Glier (Chief Executive Officer)
	- Palantir and Scale AI
	- Stanford
- Daniel Buchmueller (Chief Technology Officer)
	- AmazonPrimeAir and Vast
	- University of Zurich BS
- Brian Ballard (Chief Product Officer)
	- UpSkill, DoD "Field Operations Officer"
	- Carnegie Mellon BS MS

# Mission
- Transform defense logistics from a vulnerability into a strategic advantage through ==AI-powered software tools that enable military planners to balance efficiently and effectiveness in both peacetime and conflict==.
# Our Vision
- A future where defense logistics systems are ==responsive, resilient, and adaptive==.
- Where commanders can rapidly project and ==sustain combat power precisely when and where it's needed, regardless of operational conditions==.
# Why We Exist
- ==The credibility of our deterrence depends on modernizing defense logistics==, a critical vulnerability that, if left unaddressed, jeopardizes our ability to project power against emerging threats.
- While commercial supply chains have been transformed by predictive analytics and AI, ==defense logistics remains dependent on outdated manual processes== that cannot meet the speed, complexity, and scale demanded by modern conflicts.


# Products
- ==Navigator==: Your AI-powered decision support for military logistics. 
	- By replacing hours of manual work in spreadsheets with intuitive, advanced software.
	- Navigator delivers ==real-time predictive insights== and dramatically ==accelerates resupply decisions==, ensuring operational readiness when it matters most.
	- ==Lets units anticipate supply needs and address gaps in real-time.
	- Through ==consumption projections== that:
		- learn from historical data
		- compute ongoing requirements
		- visualize supply trends
	- ... units can make proactive adjustments before shortages impact operations.
	- The system provides ==normalized supply levels== that creates a unified view of status from any input type, whether data comes from a [[Logistics Status|LOGSTAT]], [[Joint Battle Command-Platform|JBC-P]] message, an embedded sensor, or other variants of inventory and resupply requests.
- Navigator's analytical engine ==continuously monitors supply data==, ==flagging anomalies== and ==recommending specific resupply actions== based on mission parameters and operational tempo.
- Designed to provide planners with the most accurate logistics data they'v ever seen in the field.
- Our ==algorithms== and ==AI Agents== handle the computational heavy lifting, running thousands of calculations that would otherwise burden logisticians across the team.
	- ==Input, Analyze, Act, Repeat== continuously so that logisticians can focus on courses of actions and outcomes not data jockeying.
- Input
	- Synthesizes a wide range of possible inputs to understand and estimate critical consideration factors
		- Current supply levels
		- Real-time consumptions rates
		- Realistic resupply timelines
		- Mission-based priorities
		- Operational risk factors
	- By ==aggregating and normalizing data from disparate sources==, the system creates a single, accurate picture of the logistics landscape.
- Analysis
	- Navigator's ==agents== automatically model ==possible resupply allocations and distribution plans== based on current conditions and constraints.
	- These models are generated in advance of [[Logistics Synchronization|LOGSYNC]] meetings, ==providing stakeholders with data-driven options to consider==, rather than starting from scratch. This approach transforms planning sessions from lengthy debates over basic facts to focused discussions on strategic decisions.
- Act
	- Navigator ==automates the creation of orders related to supply actions==, removing hours of wasted admin work while increasing accuracy and standardization.
	- As these actions are executed, ==their status feeds back into the systems' ground truth understanding, creating a virtuous cycle of increasingly precise logistics intelligence==. This closed-loop approach ensures that each operation improves future planning accuracy.

![[Pasted image 20260514211601.png]]
Above: 
- [[Logistics Package|LOGPAC]] is logistics Package, a tailored, synchronized convoy used by the US Army to deliver supplies, such as fuel/ammunition/rations from a Bridge Support Area (BSA) to forward-deployed units.
- [[Logistics Status|LOGSTAT]]: A military status report used by Army units to forecast sustainment requirements, coordinate resupplies, and track combat readiness.
- [[Logistics Synchronization|LOGSYNC]]: The process/meeting/briefing used to align logistics support with the operational plan. Used for coordinating fuel/ammo/food/water/repair, transportation and convoy movement, maintenance and recovery, medical support and evaluation logistics, supply priorities by unit or phase of operation, shortfalls, risks, and resupply timelines. It's where logistics and operations staff makes sure that sustainment can actually support the mission, and what constraints could break the plan.


# Use cases:
- ==Accelerated Resupply Planning==
	- Build for logisticians to plan complex resupply operations in minutes, not hours.
	- Analyzes historical consumption data, current inventory levels, and operational tempos.
	- ==Recommends optimal resupply schedules and quantities==, ensuring forces maintain peak readiness without excess inventory burden.
- ==Mission Simulation and Wargaming Integration==
	- Designed to enhance [[Tactical Table-Top Exercises]] (TTX) and Wargaming functions, bringing logistics realism into operational planning.
	- Teams can simulate scenarios to identify potential bottlenecks and vulnerabilities across the entire supply chain.
	- Seamlessly integrates into existing TTX and Wargaming workflows, letting commanders:
		- ==Stress-test logistics plans== under realistic contested environment conditions
		- ==Introduce dynamic supply chain disruptions to train resilient decision-making==
		- ==Evaluate multiple [[Course of Action]] (COA)== with complete logistics considerations
		- ==Capture== decision point and logistics ==lessons== learned for after-action review
	- Navigator's AI evaluates alternative routes, timing options, and resource allocations in real-time, during exercises, ==helping commanders select the most resilient resupply strategy while training staff to anticipate and overcome logistics challenges== before they arise in actual operations.
- ==Adapt to Disruptions==
	- When supply routes are compromised or consumption patterns suddenly change, Navigator provides immediate alternative options.
	- The system ==continuously recalculates optimal solutions based on real-time intelligence==, empowering leaders to make informed decisions in rapidly evolving situations.
- ==Data-Driven Inventory Optimization==
	- Navigator transforms each logistics operation into a valuable data asset for future planning.


_____________

Video: [Palantir for Builders | Deploying into Maven Smart System ft. Gallatin AI](https://www.youtube.com/watch?v=jK5k9_Gql-I) (April 2026)

> Q: Tactical Resupply?
> A: Best interactions between brigade/battalion (1000s of soldiers), they need to eat, drink, thousands of rounds of ammunition, building supplies, fuel. We make the process of sensing what are the supplies that they are low on... we create course of actions for the [[S4 Officer]] (Logistics Officer), and plan convoys/dispatch convoys to resupply.
> It sounds very simple, it's the most foundational set of processes that exist today, and its' done very very manually today. We've started to automate some of these manual processes with software. 
> Q: So the software connects the demand signal from the frontline and giving it back, gets the convoys moving and getting troops resupplied as needed. 
> A: Yeah, you don't win a war with logistics, but you certainly lose them if you don't have your act together.


> Q: How did you get started with your cofounder?
> A: We have 3 cofounders, Woody (CEO), Daniel (CTO), Bryan (CPO). We all come from different backgrounds. Woody was at Scale/Palantir, Bryan had his own startup that exited in the AR/warehouse optimization space, and I (Daniel) cofounded Amazon Prime Air... that was a long time ago, but since then build engineering teams at deep tech companies ranging from teledriving, self-driving, more drone delivery... and most recently at Vast space, where we're building a commercial space station in [[Low Earth Orbit|LEO]]. 
> I was introduced to woody in 2024 by [[8VC]]... and it's an important problem to solve. It's a problem that's sort of unsexy. AI has been used in intelligence and in [[Command and Control|C2]], but logistics is really left in the dust; it's always a step-child. 


> Q: Why are you building your startup on [[Palantir Foundry]]?
> A: Before cofounding, I had heard of Palantir but didn't know about it as a provider. When you deploy into mission-critical environments such as anything in defense, obviously, but other fields as well... you really need a cloud provider where... from day one, you can be assured that what you're doing comes with the required safeguards (often this is in the mil requirements), things like can you be [[DoD Impact Level 5|IL5]] compliant. ==Foundry got us to serving our customer much faster==. For a startup, everything is about speed. Once you're there... how can you iterate quickly? We release on a weekly or every other week basis into [[Maven Smart System]], and so that's super exciting for us to be able to deploy to our customers in hours, rather than days or weeks.
> Q: So you're focused on the outcomes, and Palantir gets you there faster. 
> A: Yes, also Palantir is a huge company... and we're small, about 40-45 people right now, up from scratch in 19 months. We have a joint slack channel with you all, and there are 120 people in our joint slack channel... so in a long-winded way, the attention to feedback from Palantir is unmatched. We complain a lot, find a lot of bugs, etc. And you guys care to hear feedback and quickly implement fixes (sometimes it's a P0 ticket we triggered). It feels like being part of building the product together.


> (Demo Begins)
> ==Navigator== deployed into [[Maven Smart System]]... looking at an unclassified demonstration.
> Demo unit is the 3rd Infantry Brigade Combat Team (IBCT)
> In the Logistics Common Operating Picture, you see at one glance, how we're doing. 
> To give a little bit of guidance, the color-coding is totally standard in defense logistics; You can see classes of supplies... green means good, yellow (you can set threshold) is around 60%, and red is below that, and black is... you're basically in trouble.
> You're generally trying to optimize... that a unit has "Days of Supplies" of roughly three days. You basically want to make sure that with the current supplies, they can sustain themselves for three days.
> What we can do is dive into a specific unit (e.g. the 3rd Squaron, 4th Cavalry Regiment)... and we can see an overview of that unit's ==inventory-on-hand==, what they expect to have in 24h, 48h, and so on. These are predictions from Navigator, and can be overwritten by a human at any time.

But let's play through a scenario:
- We heard over the radio that their MREs got totally wiped out.
- We save this, and then... we regenerate ==Courses of Actions== (usually done over actions by a team), and it happens nearly immediately here. We have to resupply them!
- We go to our LOGSYNCH Matrix ([[Synchronization Matrix|Synch Matrix]]), which is a standard tool to understand what the movement of resupply convoys... usually you hold a LOGSYNCH meeting twice a day. 
- Our users love using this in war-gaming scenarios; you can simulate time, so that it's not the real time showing here...
- We can see the load plan, which shows that we're currently at 0 MREs, we're going to resupply them with 257, and then they'll be at 257... this breaks down, and you can see what the convoy is made up of... and you can see the utilization of the actual underlying "prymovers" (?) transporting assets...
- In the case of autonomous convoys, we can do this automatically. 
- In KODIAK during an exercise in Hawaii at Schofield barracks, we demonstrated the whole resupply workflow from sensing that there is a shortfall at a downrange moving, determining a course of action, and dispatching an autonomous F150 truck... fully automated, end to end.

Q: So I can listen for radio traffic, look for signals, understand what that means, understand what that means, understand the overarching orders that I have, and the importance of the resupply and different components... I can crate convoys, and dispatch them to get the resupply out to groups in the field, all coordinated through one piece of software.
A: Ye, it's early days in terms of autonomous resupplies, but we very well understand that if you send out convoys without needing humans on board, it's much safer overall if you can send out (instead of a column of a convoy) a swarm, more autonomously... you have a higher survivability of supplies reaching the destination.


Q: It's pretty interesting to see... the unsexy things... that are so important to have all the right supplies at the right place at the same time.  It comes back to... the same concept of "I have to have the right stuff at the right place at the right time." It's neat that you're adapting all these things to meet people where they are.
A: We're interested in building beautiful software at Gallatin, fast, responsive, etc.. Everyone is used to amazing software, and we just feel that the warfighter deserves the same in terms of defense logistics. This tactical resupply, ==this is just the start.==  There is so much more that is broken that we feel that we can fix.

Q: What else are you excited about, broadly?
A: Interested in Computer Use Agents (CUA), and the asynchronous nature... ? ... is what really fascinates us. We're experimenting with that... what you introduced (Palantir) with long-running agents in Foundry makes a lot of sense. A lot of the time, we're already seeing that. You have agentic workflows that DO take in the dozens of minutes, because they're chains of agents working on a resupply solution, for example. We talked about supply signals earlier... you want to make it as easy as possible for the warfighter... the best [[Logistics Status]] is no Logstat, meaning it's automatically-sensed, so we're working on integration with fuel levels. Fuel is still leveled with dip-sticks in bulk (?) fuelers... So someone measures how high the wetness is and reports it. We're working on.... starting to build an need-to-send feedback cycle that doesn't need human input for basic consumption signals. 

Q: Comes back to the OODA loop, right? 
A: That longer-running agent piece is so important too. It frees up the warfighter or executive to be strategic, and the agents are doing the tactical work... it's the 1+1=3 math that we're looking for, it's a stepwise change in how we operate.


______________

[Video: Code in Production: Gallatin x Observability](https://www.youtube.com/watch?v=iKiNqLBg_Rs)  (July 3, 2025)


Host: We're not only going to be showing off a production OSDK web app on Palantir with Foundry Backend, but also a set of observability features in Foundry as we expand our capabilities as a production backend.
- Daniel will demo this @ Gallatin

Daniel Bunchmueller, CTO + Co-Founder Gallatin
- At Gallatin we believe that the AI revolution in the military isn't complete; huge proliferation in Cyber, Intelligence, C2 ... but logistics is left in the dust. We started in July 1, 2024, 20 people now, funded by 8VC.
- ==Navigator== is our primary product, built on Foundry, chose for:
	- Speed: To use premiere models in [[DoD Impact Level 5|IL5]] and [[DoD Impact Level 6|IL6]] for Top Secret environments in production. Approval to operate (speed to get approval) matters to use; Foundry comes with about 80% of the controls we need.
	- Security: Leverage the security/deployment architecture trusted by Palantir's most critical/sensitive customers.
	- Impact: Delivering transformational capabilities directly to warfighters.


![[Pasted image 20260514201536.png]]
[[Palantir Foundry|Foundry]] on the Backend
Frontend is VueJS and Typescript
We make calls into [[Palantir Ontology|OSDK]]; this is very common.

This is the view of an army  [[S4 Officer]]...
- One of the core workflows in logistics is [[Logistics Status|LOGSTAT]], logistics reports that are text, excel files, or communications over the radio.
	- "I need more 5.56"
	- We built a LOGSTAT understanding framework to parse...
		- Today, if you think about logistics, it's very manual; they get reports, don't have accurate supply level information, etc... it takes hours to get through.


> Uploads a LOGSTAT file, a pure .xls excel file.

Parses it, uses an LM to extract it, compares it against the extracted date... and says "Hey, at this time, for this unit, what's the likely consumption that will happen or has happened since then?"
- We see that we have 6k gallons on-hand... Navigator thinks a better suggestion will be 10,000. You can accept this or make a modification to it, etc.
- We can tie an NSN ([[National Stock Number]]) from literally a company level all the way up to strategic planning levels, becuase we find that there's a huge catalogue of official NSNs, so if someone says "MREs," we can tie this up with a specific National Stock Number.
- This informs the logistician of how a unit consumes goods.

So we go to the 2/27 (2nd battalion, 27th infantry regiment), and we see:
![[Pasted image 20260515004231.png]]
We see the reported values, some graphs showing both historical as well as predicted consumption.
- Why is predicted consumption relevant? It's super difficult, if you think about it.
	- Depends on lots of factors


But behind every military operation is an [[Operational Order]] (OPORD)

We can extract information like weather.. in specific  phase of operation you might be offensive, defensive, building up a post, etc... could be warmer than usual right now, etc. All of this flows into our consumption model.
![[Pasted image 20260515004728.png]]
As a commander, you have the ability to say: All of these units on the map have certain supplies and need certaiin resupplies
- Our algorithm solves both an ==optimization problem== and a ==traveling salesman problem== to say:
	- I have these trucks available, these resources to transport... 
![[Pasted image 20260515005004.png]]
Heres what we expect them to have before and after

![[Pasted image 20260515005012.png]]
Here's the proposed convoys over the next 72 hours

![[Pasted image 20260515005028.png]]
Once you approve, you can see those convoys planned out.

![[Pasted image 20260515005044.png]]
This route takes into account friendly and enemy territory, route planning based on what's actually reachable...
- Might have wading depth for a certain vehicle
	- This is specified in an open data model; we can understand this and route or propose a round accordingly.

==The decision is also with the operator to make the final decision, which creates a feedback loop to improve our algorithms.==

Now, going into a demo... 

![[Pasted image 20260515005233.png]]
This is a workflow builder... basically the entire flow of our app in one view
We've understood production... but what happens in terms of logs? That's what we're talking about today.

Host: The complexity of your backend here is very visualized here...

Yes... We can see all actions...

So let's dive in! These are the production logs here...
Let's dive in and see why it's slow
![[Pasted image 20260515005340.png]]

![[Pasted image 20260515005400.png]]
We can see that there are hundreds of object loads in the background
- Every time your function is run, you get a request log emitted to this view ... and you can dive in and... this is a function loading objects to the ontology. This is auto-instrumented for Daniel + Gallatin out of the box.















