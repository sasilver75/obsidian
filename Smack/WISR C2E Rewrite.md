

We want to have a template process for how to add a new workflow.
You could imagine that we would want to create this template process and dogfood it on the C2E rewrite.


Threat weight map: Shows what "costs" are in the network. Expected target locatoi nerror, etc.
Wizard ETL activity: principall the interaction with te ggraph, turns it into tables that Eli uses
Wizard precompute activity: This is the interaction with smack-sensors. When we precompute. For ACMs and TAIs and for sensors/targets... that full permutation, join, what's the expectdd target locatino error. IF I put a sensor in the sy and have a targe here, what's the expected TLE? That serves...



TWM: 
- It's.. not rocket science. It's saying: Im going to interrogate the graph to understand, through the weaponnet, for Blue platforms, which Red platforms can shoot stuff at me... and look at the Athena laydown and say "This T55 is sitting here, and we've decomposed the world... into a game board of some sort. This is an opinionated stp. These dots are just... Eli has ways to auto-generate the dots (Dilaney-Prune thing). If you imagine this as a fully connected network, what I do is ... wat's the range from the edge to the platform, and assign a cost to the edge based on a very simple equation...which is 1-range to edge/max effetive range. If you're closer to the edge, you get ahigher number, always less than 1,"
ETL
- One of th things that tkills us. Eli works in DataFrames. His model API... the way he thinks about it, is that he's expecting JSON blobs that can be reconstiuted into data frames at runtime, and he's expecting a very specific schema. Sometimes he does some logic to fill some rows, and sometimes he expects those columns to be filled. That's one of the biggest friction points: As eli is modeling the problem from scratch, he thinks about OR in terms of these tables, and we have to build those tables. If we he updates his conception of the model, sometimes that can change the table schema. If I want to include A2A refueling, that's a new table schema, which propagates up the stream.
- There's layers to our disfunction, one is communication.  The pipeline was not b uilt with obserbability in mind. WEve turned something that should not have been a black box into ablack box. Logging 101... Eli will say "Okay, we ned this new thing", or "The TLEs are coming back weird," adn we don't have the observabilty to say what's happening.
- The communication problem... It needs people like JC and me to sit between scientists, etc. The App eng folks are focused on API contract satisfaction and... delivering something that doesn't break, but if you say that TLE sensors.
- What do OR models use in production? Is it numpy arrays? What doest he big boy version look like... sometimes you'll see persistence of massive dataframes of DFs to Json to S3 to Temporal Workflows, which have a memory cap on what they can transmit... Some of that might be required... but maybe we need to remove from temporal entirely.
- Basically it's "Pull data from the KG"
Precompute
- I was pleased with how this turned out, Ryan hates it. Right now it does a batch process, taking all the sensors, shooters, TAIs, and ACMs. 
	- TAIs are places we shoot from
	- ACMs are places we sense from, in this context.
	- We've bastardized. Sensors hould be able to sense from any waypoint. we use ACMs to force places that things can look from, which reduces the complexity of the problem.
	- If we have 5 platform types, for evry ACM, what's their achieved TLE for every platform type in the AOI... Lots of ACMs are so far from TAIs that they shouldn't even be considered. The TLE equations don't cap anywhere... First and foremost, it's a greedy, brute-force solve... I'm going to compute the full combinatorics on this join. The second issue... which isn't one I watndt you to touch is that hte quations used to extract TLE is not that vigorous. That's  physicist problem, and Eli/JC will tell you about it... the only physics model every called in production is smack-sensors, and we wrote it, and that's ahuge problem. It's basically a ChatGPT 5.2 generataed markdown files that pass the sniff test for us. It doesn't account for weather, for isntance. Cloud cover, etc.
	- Precompute gives you: If I put a sensing platform at this ACM lookign at a target type in an ACM, this is what I wuld expect my TLE to be... and that's based on the center of mass (area) of those two geometries. Some of these are big boxes (ACM, TAIs), so that's 
		- All muntiions have: Launch TQ, INflight TQ, Terminal TQ... I want to do away with these, becaue the concept of an inflight and termianl TQ is contingent on it being able to accept udpates, and ... the US has like three munitions in the inventory that can do that. but it's very common in anti-ship cruise missiles. Really, we should probably be doing launch CQs and call it. that would be onrmal
Solve
- Eli gets all his dataframes, and pushes them or bumps them up to a Gurobi solver.... HE initializes a slve object, it's just an equation, a simplex algorithm...
- The terms in the equation represent constraints as he's defined them... thingsl ike "can this thing acheive TQ, does the platform hav enough of the necessary munitions ot shoot the thing," etc. I think eh's also optimizing for risk, and maybe endurance as well. 
- Black box: Deliver him the data frames and constraint requirements, and he returns a solved artifact, which gets converted to our GANT.
	- I don't want you to worry about this part: Can you reflect his solve backo n the UI... We've failed to do that, in some respects. This is an app eng thing, not you.
Store Solve
- WE're just persisting artifacts


WE know that we've done it oday in a way that's hard to monitor, hard to observe, and probably more complicated than it needs to be.
It's probably also inefficient, doing joins we don't need to do, having high memory artifacts that don't need to be persited or saved.
The other thing this pipleine is not.. we've gne over a few examples today of ways that we would increase the compleixty:
- GPs being autogenerated by obsering google maps terrain data
- WP generation including a 3D grid (altitude)
I don't think the current pipeline is very agiblee to iterative ocmplexity either.
IF Eli were to say: "I want to include altitude," we're not well-postured to layer that on to this pipeline
In the grand scheme... John's thoughts: We very likely have two decoupled things:
- Model versionining
- Pipeline versioning
We're seeing that today: The NEC model used to compute this solutoins:
![[Pasted image 20260826124227.png]]
Is so substantively different from our legacy one that if you were e to deploy this onto our stack and try to do the south china sea scenario, it would break.
- This NEC version... got pushed from dev up into demo.smackdev... and then Marti made some changes the original OR model adn pushed it back through up to.... we have basically divergent code states. 
- WE would like... what Ari calls the "Model Stable" Maybe it's in the UI, maybe it's not, but... we would want to be able to select which model we invoke at any giventime.
- As we broaden to logistics, etc. Those would be distinct models hat might be able to resuse a pipeline
	- e.g. the NEC and original one should reuse the same pipeline (and should be the same model, because it's the same usecase)... that shouldn't be a divergent model... if we get to new use cases... we're going to be need to be able to version these things.
- Eli right now... thisversion doesn't allow Type 12s (ground) to rearm, for instance. The NEC model 2 might do that, and its pieline could be updated to support it. We might want to retain the old model... and maybe developing a stable of versioned models that are capable of doing different things.
- A great way of articulting taht: The next iteration of arearm is basically having adelay (they can't be scheduld within 2 hours of shooting), but going to v3 would be a secondary logistics optiization that says "Here's our logistcs supply points ,etc." Our pipeline rn is hard to understrand, people don't like it, think it's bad, and it's always a chore to repair this thing.
- This pipeline ... has shifting API boundaries both in and out. Graph architecture is updating, so it's possible that cypher traversals that used to work need to be updated because the architecture got more complicated... And obvious Eli's requirements change. This thing kicks our ass. In demos, we say that our approach can be applied across warfighting functions... but if we look clear-eyed internally, we don't have the processes in place to really rip that in a reasonable amount of time, we'd probably do a one-off brittle solution. That's C2E in a nutshell.

I think there's always graphs... There's a lot of paths to the concept that could e expanded upon. THe most imediate is altitude. We're currently in a two-dimensional artifact. If I'm flying very high as a B-52, I shouldn't incur risk. It's coool, it's got potential, and we've just not refriend it since.
We'd like to do this type of modeling for things like weather, electromagnetic interference. You can also weight transits by fuel consumption, etc. So you have endurance and efficicacy.

Expected approach:
- Start by familiarizing myself with what exists today.
	- Informaiton in PLane, which you don't havaccess to
	- Gitlab
- Eli JC and Sam sitting in a room asking" What hshould the proces be?"
	- Going from nothing to model that does X
	- For 82nd airborne... we sat in a room with JF and understood a problem, translated the spec into modeling constraints, stepped up, agreed on API boundaries, etc.
	- WE want to execute this against C2E


Eli... it's like a shakespearaean tragedy. That guy knows how to wield OR to do some really sexy things, but we as a company have never acknowleged that we want to do that. This is where GenX comes from, an RL warm start to an OR problem. Allows RL to enter the solution space, and OR to explore that solution space in an efficient way. Eli got his from a paper in 2025, pretty bleeding edge. We haven't operationalized that approach or things like it. The RL/OR hybrid model... that approach...


TWM, ETL, and PRecompute can be done as soon as you upload geometries, 

GenX was 





_____________

Sam Observations August 26, 2026
- Note that the P2C generation pipeline (which I'm not looking at directly) seems to have more recovery controls in contrast to the C2E pipeline that I'll actually be looking at (which doesn't have the same controls):
	- A persisted work claim
	- Heartbeats
	- Stale-claim takeover
	- Deterministic stage workflow IDs
	- Stage-aware recovery
	- Database locks
	- A check that the expected workflow run is still current, before committing results.
- Once the user approves a generated Concept of Fires in (5. Review), behind the scenes, the frontend calls Maestro, which verifies that (The concept of fire belongs to the selected `coa_workflow`, Required FST output exists, Required target output exists). Then, the user globally approved the ConOp for the Unified product, after clicking through (6. Workbench) and pressing "Approve" on the ConOp, back in the main screen.
	- When this happens, Maestro updates `atlas_conop_group.approved_conop_id`, deletes and recreates group-scoped Track rows from the approved targets... 
- User the ncreates a C2E Task by entering/submitting a task name and start time, and submits it
	- ![[Pasted image 20260826141638.png]]
	- The browser: 
		- Generates an `operationId` using `crypto.randomUUID()`
		- Creates a local pending opeation.
		- Calls Maestro's `startWizardSolve(...)` mutation.
			- Despite its name, the `startWizardSolve` does ==not== start Wizard or Temporal, it just results in inserting an `(operationid=browserUUID, conop_id=approved_workflow_id, status=pending)` record to the `atlas_wizard_solve` table. No temporal workflow exists, and no model work has started.
		- Navigates to `/unifeid/c2e` after the mutation succeeds.
- The user now configures the model input:
	- On the C2E screen, the user selects targets, configures assignments (this "edit" dosen't work now, for noted bug reasons), and clicks generate.
	- When generate is clicked, the frontend assembles wizard input from:
		- Approved workflow/conop ID
		- The workflow group id
		- The browser-generated operation ID
		- Geometry data
		- Approved targets
		- Weapon or resource assignments
		- Other model parameters
	- It loads, merges, and validates the required geometry before sending the request.
		- ((I wonder what it's actually sending... is it appropriate that it be sending actual geometry data, for instance, as opposed to identifiers for such things?))
	- Frontend then calls `startWizardOrchestrationWorkflow`.. This actually initiates the model pipeline
- Maestro records "processing" before it starts Temporal
	- Behind the scenes, Maestro receives the workflow ID, group ID, operation ID, and model input as independent values...
		- It verifies the group exists, but doesn't prove that the workflow belongs to the group, the operation belongs tot he same logical run
	- Maestro writes atlas_wizard_solve.status = processing and temporal_workflow.status = pending
	- It schedules a process-local `setImmediate(...)` continuation and returns success to the frontend.
	- The request succeeded and the operation appears to be processing.
- Temporal starts the Wizard workflow:
	- Maestro derives a deterministic Temporal workflow ID: `wizard-model-orchestration-<operationId>`
	- Temporal records the durable workflow history and sends activities to the Wizard task queue...


In the UI, we see:
- Genreating Threat Weight Map (fast)
- Structuring Model Inputs
- Computing Model Inputs
- ...?
- Running Solver


The checked-in `demo`, `staging`, and `gov` configurations point Wizard at `https://solver.smackdev.com`, and the Wizard request model defaults to remote solving. In that path, Wizard serializes the locally built Gurobi model as a compressed MPS file, posts it to the remote solver, and waits synchronously for the solution.

So the expensive mathematical optimization is intended to run in a separate solver service, but the Wizard activity slot remains occupied for the entire request. The slot is not released while Wizard waits on the remote HTTP call. A 20-minute remote solve therefore blocks TWM, ETL, precompute, solve, and result-storage activities for every other run assigned to that worker.

In short: **Wizard builds and processes the model locally; the remote service performs the optimization; Wizard’s only activity slot remains occupied while both happen.**

The Wizard worker is a long-running Python process inside the Wizard K8s container.

```
Wizard Kubernetes pod
└── Python entrypoint process: python -m app.entrypoint
    ├── Python health process: python -m app.health_main
    └── Python worker process: python -m app.main
        └── Temporal Python SDK Worker
            ├── polls wizard-model-queue
            ├── runs Wizard workflow code
            └── executes the five activity functions
```

Wizard isn't a web app, it's primarily a background Temproal worker. The main work doesn't arrive through an HTTP endpoint: 
1. Maestro asks Temporal to start a workflow
2. Wizard continuously polls Temporal's `wizard-model-queue`]
	- Note that Temporal manages several related thigns: Workflow executio nhistory, worfklow-task queues, activity-task queues, timers and activity timeouts, retry scheduling cancellation request ,assignment of queued tasks to polling workers.
		- Workflow Tasks/Queue vs Activity Tasks/Queues
			- Workflow tasks tell a worker to advance teh workflow state, such as deciding that the TWM activity should run next
				- "Given the workflow history so far, what should happen next?" The worker runs the workflow's orchestration code. Tasks should be short. Tehy don't perform the model computation, make HTTP requests, or write artifacts. The output is a set of cmomands returned to Temporal, such as (schedule an activity, starta time,r, record progress, complete orr fail a workflow).
				- Must be deterministic, because Temporal might replay it from history.
			- Activity tasks tell a worker to execute actual side-effecting code such as generating TWM or calling the solver.
				- Asks"Execute thsi specific piece of work": Things like `wizard_twm_activity`, `wizard_etl_activity`, etc. ACtivities perform side effects and expensive computation: Querying/transforming data, reading/writing iris objects, building urboi model, calling remote solver, calling maestro over gRPC.
			- Both can use the same-named Temporal tasks queue as they do here.

3. Temporal gives Wizard an activity task
4. Wizard executes the corresponding Python function
5. Wizard cals other services as needed
6. Wizard reports activity cmopletion back to TEmporal


Activity Tasks and Workflow Tasks, and how they alternate/relate:
1. Temporal sends workflow task
   “The workflow just started. What is next?”

2. Wizard runs workflow code
   → responds: “Schedule TWM activity”

3. Temporal sends TWM activity task

4. Wizard executes TWM
   → stores artifact
   → returns artifact reference

5. Temporal records the TWM result and sends another workflow task
   “TWM completed. What is next?”

6. Wizard runs workflow code
   → responds: “Schedule ETL activity”

7. Temporal sends ETL activity task