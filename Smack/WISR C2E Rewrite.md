

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


____________

August 27

C2E means ConOp to Execution, part of the Unified product which takes an approved operational concept and asks a model to propose when and how selected targets should be serviced.
- Maestro is the primary backend service powering Unified; validates API requests, reads/writes product data in Postgres, starts Temporal workflows ,receives Wizard results, and converts those results into Strato rows.
	- Postgres stores the product's view of the operation: pending and processing solve rows, completed result JSON, copied workflow status, and the Track/Engage board information.
- Temporal is the durable workflow engine. It records that a Wizard workflow exits, schedules its five activities in order, records retries and timeotus, preserves execution history across worker restarts. Temporal remembers/manages the workflow/pipeline, and Wizard currently performs/drives completion of all five of its activities.
- Wizard (WISR) is the Python worker that performs the C2E model pipeline, receiving work from Temporal and running:
	- Threat-Weighted Mapping (TWM) constructions
	- ETL/table extraction
	- Precomputation
	- Optimization/Solving
		- Depending on configuration, this runs locally or through a remote HTTP service. Gurobi is the underlying optimization tool used.
	- Result submission to Maestro
- Iris is Smack's provider-agnostic object-storage API used here to store large intermediate Wizard documents. Wizard writes its TWM, extracted table, and precomputation data to Iris. Temporal then carries small references to those objects instead of placing the full documents in workflow history.
- Strato is Unified's execution-planning screen, at `/unified/execute`. It calls Maestro APIs to read/modify data powering the frontend components of the board. Postgres `atlas_exstage_*` tables store its state. When Wizard finishes a solve, Maestro converts parts of the result into Strato Engage rows. An operator can also move a target between Track and Engage, change assignments, and edit certain planning values.

An approved ConOp is the plan package that already contains targets, boundaries, available forces, and other planning constraints.

C2E then adds a particular *operation*, which includes a start time and the operator's selected targets, as well as (optionally) platforms and effectors mandated to service those targets. The model then proposes timings for specific, target-paired ISR and strike platforms (along with the effectors used for the strike platforms). This output is collectively referred to as a "proposed tasking."
- ==Note:== Currently, the user-selected platforms/effectors for a given target are currently ignored by the optimizer. The dropdown options are supposed to be populated by a `getBlueAttackOptions` request (which is getting sent), but I don't often see the response containing attack options (Ref: alkemaf234), which means that the dropdowns to select an effector or platform are usually empty anyways. Aegis is intended to satisfy that `getBlueAttackOptions` request, joining currently represented BLUE platforms in the workflow-group graph, WpnNet compatibility rows about which platform types can employ munitions against which target types, and approved target instances and their platform types. 

Before the model pipeline (in Temporal) begins, Unified already has an approved ConOp. That record identifies the planning workflow whose targets/constraints are allowed to move into C2E. The operator also has created named operation, which is the Unified product's record of this particular attempt to produce a plan.
- Creating the operation writes a pending `atlas_wizard_solve`  row in Postgres, but does not start any sort of solve or model run.
- Clicking the "Generate" button is what actually kicks off work. This means that a "pending" solve/operation row is not evidence that the model ran.

Let's say that the operator asked the C2E product to plan a single strike, selecting only target `T-055_105_CHN`, and manually assigning one F-35 and an AGM-158 as its effector for that target, asking C2E to produce a feasible plan/tasking.
- Note that the result may change the exact aircraft or weapon variant (because the solve doesn't use this user-specified platform/effector information)

A solved plan can contain (at least):
- Target time
- Weapon
- Attack aircraft
- ISR platform
- ISR window
- Latest track quality (TQ)

When the operator clicks Generate, `startWizardOrchestrationWorkflow` is called, passing (non-exhaustively):
- IDs: `workflowId, workflowGroupId, operationId`
- Target Array: ID (target's identifier, such as "T-055_105_CHN"), rank (the target's position in the approved prioritized-target list), formation (the parent unit or platform associated with the target, like "T-055"), supported-FST record IDs (one-based references to the approved FST that the target supports; \[1,3] indicates that the target contributes to FST-1 and FST-3), weapon + platform selections
- 7 required geometry strings:
	- Waypoints (WP)
	- Air, Ground, Maritime Control Measures (ACMs, GCMs, MCMs)
	- Target Areas of Interest (TAIs)
	- Unit Boundaries (UBs), one of which is selected as the user and doubly sent as the Principal Boundary (PB)
- Additional, optional geometry: Restricted Areas (RAs) and Air to Air Refueling Points (AARs)

> [!NOTE]- Q: Why do we send a geometry snapshot?
> The Wizard workflows run asynchronously... by the time a later activity executes or retries, the approved geometry may have changed. The workflow therefore needs either an immutable copy of teh geometry used for this run, or references to immutable, versioned geometry stored on the server.

> [!NOTE]- Q: Why do we send geometries encoded as Strings?
> There's not really a demonstrated need for that, it's just how it's implemented. Unified uses JSON.stringify(...), and Wziard eventually calls json.loads(...) to recover the GeoJSON object. 
> Instead of:
> ```
> {
> "waypoints": "{\"type\":\"FeatureCollection\",\"features\":[...]}"
> }
> ```
> 
> It could have been sent as
> ```
> {
>  "waypoints": {
 >   "type": "FeatureCollection",
  >  "features": []
 > 	}
> }
> ```
> GraphQL doesn't require string representation, it could use a JSON scalar, typed GeoJSON inputs, or (probably preferably) immutable artifact references with digests.

So the input to our `startWizardOrchestrationWorkflow` looks *something* (I think) like:
```json
{
  workflowId: "11111111-1111-4111-8111-111111111111",
  workflowGroupId: "22222222-2222-4222-8222-222222222222",
  operationId: "33333333-3333-4333-8333-333333333333",
  operationName: "Operation 20260601T191825",
  operationStartDate: "2026-06-01T19:18:25.869Z",
  approvedTargets: [{
    targetId: "T-055_105_CHN", rank: 3,
    formation: "T-055", supportedFsts: [1,2,3],
    selectedMunition: "AGM-158",
    selectedPlatform: "F-35-01"
  }],
  waypoints: "{\"type\":\"FeatureCollection\",...}",
  acms: "{...}", gcms: "{...}", mcms: "{...}",
  tais: "{...}", unitBoundaries: "{...}",
  principalBoundary: "{...}"
}
```
Note that the geometry collections aren't database references, they're the browser's snapshot of waypoints, control measures, target areas, and boundaries at the moment Generate is clicked.

Maestro accepts this request, and tells the browser that the request succeeded, and only afterwards asks Temporal to create the durable workflow.
- Maestro validates that the group exists, marks the solve as "processing" in the database.  It returns a positive response to the browser, and *then* asks Temporal to start via an in-memory follow-up (==Danger==).

Now that the Temporal workflow is running, Wizard progressively turns this product data into an optimization result.
To do this, it executes five Wizard activities in a fixed order, with each activity transforming input data and producing data for the next one.
The stages are as follows (and their fixed "progress". The percentages shown in Unified are emitted when the activities start):
1. TWM (Threat-Weighted Map Generation), 5%
	- Combine geometry and workflow-group data into the model's initial structure.
	- Result ➡️ Iris
2. ETL (Extract, Transform, Load), 25%
	- Create 22 serialized tables for targets, assets, control measures, and counts.
	- Result ➡️ Iris
3. Pre-Compute (Compute Model Inputs), 45%
	- Prepare weapon effectiveness, sensor efficacy, targetability costs, and observations.
	- Result ➡️ Iris
4. Solve (Running the optimization/solve), 70%
	- Load extracted, precomputed data and choose timing, weapons, attack assets, and ISR supports.
	- Result ➡️ Memory
5. Store Solve (Storing Model Output), 95%
	- Serialize the result and send it back to Maestro for product-state writes.
	- Result ➡️ Maestro

The first activity builds a ==threat-weighted map (TWM)==, combining geometry strings set by Unified with workflow-group force data, producing the initial model-oriented structure used by later stages.

The second activity is the ==extract, transform, and load (ETL)== or data-shaping stage. It turns the threat-weighted-map structure into 22 serialized tables covering targets, assets, units, control measures, and row counts.

The third activity ==pre-computes== model parameters that are expensive or awkward to derive inside the optimizer. Its output includes weapon effectiveness, sensor efficacy, targetability costs, and sensor observations. The solver later reads the extraction and precompute artifacts together.

For these steps, intermediate data moves through Iris. Wizard doesn't pass each intermediate document through Temporal history. It writes the document to Iris and gives Temporal. Temporal carries a reference to each document, not the document itself.
- The reference records the bucket, object key, content type, byte count, SHA-256 content digest (a checksum) and creation time.
- The next Wizard activity actually uses only the bucket and object key to download the appropriate bytes from Iris, and parse the resulting JSON.
- Many fields inside the downloaded document are themselves JSON strings representing tabular DataFrame data.

In the fourth activity, the solver activities receives (via previous artifacts) tables answering five practical planning questions:
1. What must be serviced?
	- ID/Cost Class
2. What weapons can affect each target type?
	- Effector/TargetType/Probability
3. What sensors can observe it?
	-  Sensor/Area/TLE
4. What assets and inventories are available?
	-  Assets and Laydown (Red/blue units, aircraft locations, weapon inventory, platform capabilities, and formation relationships)
5. Time, geometry, and capacity limits for the plan.
	- Operation start, boundaries, waypoints, control measures, target areas, resource capacity, and model compatibility rules. The selected UI platform/munition strings are not constraints.

Wizard then filters to approved target IDs, converts tables to typed records, builds a Gurobi optimization problems, and invokes the configured local or remote solver.

In the fourth solver activity, the solve blocks the only Wizard activity slot until it returns (whether optimization occurs via a local solver path or a configured remove HTTP solver). The activity does not emit progress heartbeats while the optimizer is running.
- ==Note==: The activity does not emit progress heartbeats while the optimizer is running; it returns only after the solver produces its result or the call fails. Temporal can stop waiting for the activity, while the code that was performing the optimization continues to consume the worker slot. The configured worker allows only one Wizard activity at a time, one cancelled or stuck solve can delay threat-weighted-map generation, data shaping, precompute, solve, and storage activities for every other operation using that queue.

The solver then returns a proposed schedule:
- The solver returns a nested JSON object containing target schedules, custody windows, realized target quality, attack-asset tasking, travel information, objectives, and solve statistics.

![[Pasted image 20260827143118.png]]

In the fifth activity, Wizard serializes the entire model result into a JSON string and sends it to Maestro over gRPC via `StoreWizardSolveRequest`. The request identifies the ConOp, the operation, and the Temporal workflow, but it does not identify the specific Temporal run or activity attempt that produced the result.

This plan response looks like (values are only illustrative):
```json
# At the top level, we have three keys
{
  plan_response: { ... },
  extended_results: { ... },
  visualization_context: { ... }
}

# Plan Response: The proposed plan
{
  "request_id": "33333333-3333-4333-8333-333333333333",  # Correlates the result with the request supplied to Wizard.
  "response_id": "c702a9d7-01de-4da1-8551-b95e87675a7d", # New identifier generated for this exported response.

  "target_schedules": [                                  # One entry for every target the model scheduled for an effect.
    {
      "priority": 3,                                     # Target priority used in the model's target table.
      "target_number": "T-055_105_CHN",                  # Target instance identifier.
      "tot": "2026-06-01T22:49:07.792362",               # Planned time on target.

      "effector_info": {                                 # Weapon allocation selected for this target.
        "effector_type": "AGM-158B-2",                   # Selected weapon/effector type.
        "effector_count": 1,                             # Quantity allocated to the target.
        "tq_gates": null                                 # Optional legacy target-quality gate data.
      },

      "custody_windows": [                           # Selected sensor coverage periods supporting this target.
        {
          "time_window": {
            "start": "2026-06-01T20:02:28.000000",       # Beginning of the selected coverage interval.
            "end": "2026-06-01T20:59:43.000000"          # End of the selected coverage interval.
          },
          "platform_id": "MQ-9A-ER_007.VMU-1.MAG-13.3MAW_USA", # Sensor-platform instance assigned tointerval.
          "sensor_type": "RADAR",                        # Sensor used during the interval.
          "coverage_type": "TARGETING"                   # Purpose of coverage; currently TARGETING or BDA.
        }
      ]
    }
  ],

  "realized_tqs": [                                     # Target-quality values produced by the selected observations.
    {
      "target_id": "T-055_105_CHN",                     # Target whose quality history follows.
      "realized_tqs": [
        {
          "time": "2026-06-01T22:03:25.000000",          # Time at which this quality value is realized.
          "tq": 8,                                      # Model target-quality value at that time.
          "custodians": [
            "MQ-9A-ER_007.VMU-1.MAG-13.3MAW_USA"         # Sensors/platforms responsible for that quality value.
          ]
        }
      ]
    }
  ],

  "effector_tq_gates": [                                # Timing and target-quality requirements associated with weapon use.
    {
      "target_id": "T-055_105_CHN",
      "tot": "2026-06-01T22:49:07.792362",

      "release_time": "2026-06-01T22:20:00.000000",      # Planned weapon-release time.

      "cue_tq_gate_time": "2026-06-01T20:59:43.000000",  # Time by which the cue-quality requirement must be satisfied.
      "cue_tq_gate_threshold_time": "2026-06-01T20:02:28.000000", # Beginning of the cue observation window.
      "cue_required_tq": 6,                             # Required target quality for cueing.

      "inflight_tq_gate_time": "2026-06-01T22:35:00.000000", # Time by which in-flight quality must be satisfied.
      "inflight_tq_gate_threshold_time": "2026-06-01T22:20:00.000000", # Beginning of that requirement window.
      "inflight_required_tq": 7,                        # Required target quality while the weapon is in flight.

      "terminal_tq_gate_time": "2026-06-01T22:49:07.792362", # Terminal-quality evaluation time.
      "terminal_tq_gate_threshold_time": "2026-06-01T22:45:00.000000", # Beginning of terminal requirement window.
      "terminal_required_tq": 8,                        # Required terminal target quality.

      "bda_time": null                                  # Planned battle-damage-assessment time, if modeled.
    }
  ],

  "asset_travel_logs": [                                # Planned movement history for each selected or considered asset.
    {
      "asset_id": "F-35B_001.VMFA-122.MAG-13.3MAW_USA",

      "mezpen_report": {                                # Modeled threat-penetration information for the route.
        "total_duration_seconds": 0,                    # Total time spent inside modeled penetration windows.
        "pen_windows": []                               # Individual penetration time windows.
      },

      "transit_legs": [                                 # Coarse origin-to-destination movements.
        {
          "transit_window": {
            "start": "2026-06-01T21:40:00.000000",
            "end": "2026-06-01T22:20:00.000000"
          },
          "start_position": {
            "latitude": 19.0486,
            "longitude": 120.1038
          },
          "end_position": {
            "latitude": 20.8,
            "longitude": 121.0
          }
        }
      ],

      "route_segments": [                            # Step-level movements used to construct the complete route.
        {
          "time_window": {
            "start": "2026-06-01T21:40:00.000000",
            "end": "2026-06-01T22:20:00.000000"
          },
          "segment_type": "transit",                    # Movement kind, such as transit or hold.
          "move_id": "MOVE-104",                        # Identifier of the selected graph edge/move.
          "step": 4,                                    # Discrete model time step.
          "start_location_id": "GP-001",                # Origin node in the model network.
          "end_location_id": "TAI-055",                 # Destination node in the model network.
          "start_location_type": "gp",                  # Origin category; here, generation point.
          "end_location_type": "tai",                   # Destination category; here, target area of interest.
          "start_position": {
            "latitude": 19.0486,
            "longitude": 120.1038
          },
          "end_position": {
            "latitude": 20.8,
            "longitude": 121.0
          }
        }
      ]
    }
  ]
}


# Extended Results: Why and how the plan was formed
{
  "asset_taskings": [                                  # Final role selected for each asset.
    {
      "asset_id": "F-35B_001.VMFA-122.MAG-13.3MAW_USA",
      "asset_type": "F-35B_USA",
      "role": "atk",                                    # atk = attack role.
      "target_id": "T-055_105_CHN"                     # Target assigned to this attack asset.
    },
    {
      "asset_id": "MQ-9A-ER_007.VMU-1.MAG-13.3MAW_USA",
      "asset_type": "MQ-9A-ER_USA",
      "role": "isr",                                    # isr = intelligence/surveillance/reconnaissance role.
      "target_id": "T-055_105_CHN"
    }
  ],

  "threat_cost_summary": [                             # Modeled objective cost incurred by asset routes.
    {
      "asset_id": "F-35B_001.VMFA-122.MAG-13.3MAW_USA",
      "total_threat_cost": 4.8846,                     # Sum of modeled threat cost across this asset's route.
      "route_segments": [
        {
          "move_id": "MOVE-104",
          "threat_cost": 4.8846,                       # Threat-cost contribution for this movement.
          "incurred": true                             # Whether this cost was charged to the selected solution.
        }
      ]
    }
  ],

  "isr_opportunities": [                              # Feasible sensor opportunities that were not selected.
    {
      "asset_id": "MQ-9A-ER_008.VMU-1.MAG-13.3MAW_USA",
      "target_id": "T-055_105_CHN",
      "sensor_type": "RADAR",
      "start_hr": 1.2,                                # Start time in hours relative to the planning-window start.
      "end_hr": 2.1,                                  # End time relative to the planning-window start.
      "tle_km": 1.5,                                  # Expected target-location error in kilometers.
      "reason_not_selected": "A lower-cost custody opportunity satisfied the requirement"
    }
  ],

  "objective_breakdown": {                            # Decomposition of the optimization objective.
    "total_objective": 187.25,                        # Sum of the objective contributions below.
    "pk_contribution": 200,                           # Reward associated with modeled probability of kill/effects.
    "threat_cost_contribution": -12.21,               # Penalty from threat exposure along selected routes.
    "isr_coverage_contribution": 0,                   # Legacy ISR objective field; configured implementation reports zero.
    "timing_penalty": -0.54                           # Penalty associated with target timing.
  },

  "solve_statistics": {                              # Statistics reported by the mathematical solver.
    "status": "OPTIMAL",                              # Solver termination status.
    "objective_value": 187.25,                        # Solver's value for the selected solution.
    "mip_gap": 0,                                     # Gap between the solution and best bound; zero means proven optimal.
    "solve_time_seconds": 18.42,                      # Time spent solving the optimization problem.
    "num_variables": 18432,                           # Number of decision variables in the model.
    "num_constraints": 27106,                         # Number of mathematical constraints.
    "num_iterations": 964                             # Solver iteration count.
  }
}

# Visualization Context: Bounded diagnostic data
{
  "version": 1,                                      # Version of this diagnostic/visualization schema.

  "network": {                                       # Spatial network and feasible decision space.
    "locations": [
      {
        "location_id": "TAI-055",                    # Model location identifier.
        "location_type": "tai",                      # Location category: target area of interest.
        "name": null,                                # Optional human-readable name.
        "latitude": 20.8,
        "longitude": 121.0
      }
    ],

    "candidate_edges": [                             # Every movement edge available before optimization.
      {
        "move_id": "MOVE-104",
        "move_type": "air",
        "start_location_id": "GP-001",
        "start_latitude": 19.0486,
        "start_longitude": 120.1038,
        "end_location_id": "TAI-055",
        "end_latitude": 20.8,
        "end_longitude": 121.0
      }
    ],

    "threat_edges": [                                # Candidate edges that carry modeled threat cost.
      {
        "move_id": "MOVE-104",
        "asset_type": "F-35B_USA",                   # Asset type for which this threat calculation applies.
        "move_cost_targetable": 0.1954,              # Raw targetable-threat cost for the edge.
        "num_threats_targetable": 2,                 # Number of modeled targetable threats affecting the edge.
        "start_location_id": "GP-001",
        "start_latitude": 19.0486,
        "start_longitude": 120.1038,
        "end_location_id": "TAI-055",
        "end_latitude": 20.8,
        "end_longitude": 121.0
      }
    ],

    "chosen_moves": [                                # Movement decisions selected by the optimizer.
      {
        "asset_id": "F-35B_001.VMFA-122.MAG-13.3MAW_USA",
        "asset_type": "F-35B_USA",
        "step": 4,
        "move_id": "MOVE-104",
        "move_type": "air",
        "move_time_hr": 0.67,                        # Modeled duration of the movement.
        "move_distance_km": 210.4,
        "xmove": 1,                                  # Binary decision value; 1 means the move was selected.
        "start_location_id": "GP-001",
        "start_latitude": 19.0486,
        "start_longitude": 120.1038,
        "end_location_id": "TAI-055",
        "end_latitude": 20.8,
        "end_longitude": 121.0
      }
    ],

    "feasible_strikes": [                            # Strike combinations available before optimization.
      {
        "asset_type": "F-35B_USA",
        "target_type": "T-055",
        "effector_type": "AGM-158B-2",
        "standoff_traversal_hr": 0.4,                # Modeled weapon travel time from launch point to target area.
        "start_location_id": "WP-055",               # Candidate launch waypoint.
        "start_latitude": 20.2,
        "start_longitude": 120.6,
        "end_location_id": "TAI-055",
        "end_latitude": 20.8,
        "end_longitude": 121.0
      }
    ],
    "feasible_strikes_total_count": 1,               # Total feasible strike rows before output-size limiting.
    "feasible_strikes_returned_count": 1,            # Number actually included in this JSON.

    "feasible_isr": [                                # Sensor/target combinations available before optimization.
      {
        "source_type": "organic",                    # Organic platform rather than satellite.
        "asset_type": "MQ-9A-ER_USA",
        "target_type": "T-055",
        "sensor_type": "RADAR",
        "tle_km": 1.2,                               # Expected target-location error.
        "start_location_id": "ACM-009",              # Sensor operating/control-measure location.
        "start_latitude": 20.1,
        "start_longitude": 120.7,
        "end_location_id": "TAI-055",
        "end_latitude": 20.8,
        "end_longitude": 121.0
      }
    ],
    "feasible_isr_total_count": 1,
    "feasible_isr_returned_count": 1
  },

  "sequence": {                                      # Time-ordered decisions in the selected plan.
    "planning_window_start": "2026-06-01T19:18:25.869000",
    "horizon_hr": 24,                                # Total modeled planning horizon.
    "num_steps": 12,                                 # Number of discrete model time steps.

    "asset_steps": [                                 # What each asset does during each model step.
      {
        "asset_id": "F-35B_001.VMFA-122.MAG-13.3MAW_USA",
        "role": "atk",
        "step": 4,
        "start_hr": 2.37,                            # Step start relative to planning-window start.
        "end_hr": 3.04,
        "release_hr": 3.03,                          # Weapon-release time, when applicable.
        "observation_hr": null,                      # Sensor observation time, when applicable.
        "action": "air",                             # Selected movement/action type.
        "is_hold": false,                            # Whether the asset remains at one location.
        "move_id": "MOVE-104",
        "xmove": 1,
        "start_location_id": "GP-001",
        "start_latitude": 19.0486,
        "start_longitude": 120.1038,
        "end_location_id": "TAI-055",
        "end_latitude": 20.8,
        "end_longitude": 121.0
      }
    ],

    "target_outcomes": [                             # Selected and unselected outcomes for model targets.
      {
        "target_id": "T-055_105_CHN",
        "target_type": "T-055",
        "target_location_id": "TAI-055",
        "cost_type": "targetable",                   # Threat/target cost category.
        "priority": 3,
        "target_class": "principal",                 # Principal target or residual target.
        "residual": false,
        "struck": true,                              # Whether the optimizer selected an effect against it.
        "status": "struck",
        "smack": 1,                                  # Underlying binary target-selection variable.
        "tot_hr": 3.51                               # Time on target relative to planning-window start.
      }
    ],

    "strike_events": [                              # Detailed selected weapon-employment decisions.
      {
        "asset_id": "F-35B_001.VMFA-122.MAG-13.3MAW_USA",
        "asset_type": "F-35B_USA",
        "step": 4,
        "target_id": "T-055_105_CHN",
        "target_type": "T-055",
        "effector_type": "AGM-158B-2",
        "ae_ct": 1,                                  # Number of effectors allocated by this asset.
        "strike_spec": 1,                            # Binary decision indicating this strike was selected.
        "release_hr": 3.03,
        "tot_hr": 3.51,
        "start_location_id": "WP-055",
        "start_latitude": 20.2,
        "start_longitude": 120.6,
        "end_location_id": "TAI-055",
        "end_latitude": 20.8,
        "end_longitude": 121.0
      }
    ],

    "observation_events": [                         # Selected ISR observations supporting weapon requirements.
      {
        "requirement_id": "T-055_105_CHN:cue",       # Identifier of the ISR requirement being satisfied.
        "phase": "cue",                              # Cue or in-flight phase.
        "source_type": "organic",                    # Organic platform or satellite.
        "observer_id": "MQ-9A-ER_007.VMU-1.MAG-13.3MAW_USA",
        "observer_type": "MQ-9A-ER_USA",
        "observer_step": 2,
        "observer_location_id": "ACM-009",
        "target_id": "T-055_105_CHN",
        "target_type": "T-055",
        "target_location_id": "TAI-055",
        "effector_type": "AGM-158B-2",
        "sensor_type": "RADAR",
        "sensor_tle_km": 1.2,
        "required_tq": 6,
        "observation_hr": 1.4,
        "window_start_hr": 0.73,
        "window_end_hr": 1.69,
        "cue_req_indiv": 1,                          # Binary decision indicating this cue opportunity was selected.
        "strike_spec": 1                             # Links the observation to a selected strike.
      }
    ],

    "skipped_approved_targets": []                  # Approved targets excluded before/during model construction.
  },

  "inventory": {                                    # Available resources and selected allocations.
    "platforms": [
      {
        "asset_id": "F-35B_001.VMFA-122.MAG-13.3MAW_USA",
        "asset_type": "F-35B_USA",
        "home_location_id": "GP-001",
        "role": "atk"                                # Role selected by the model.
      }
    ],

    "selected_loadouts": [
      {
        "asset_id": "F-35B_001.VMFA-122.MAG-13.3MAW_USA",
        "asset_type": "F-35B_USA",
        "loadout_id": "LOADOUT-AGM158-2",
        "xload": 1                                   # Binary decision indicating this loadout was selected.
      }
    ],

    "effectors": [
      {
        "asset_id": "F-35B_001.VMFA-122.MAG-13.3MAW_USA",
        "asset_type": "F-35B_USA",
        "effector_type": "AGM-158B-2",
        "capacity": 2,                               # Weapon capacity available on this asset.
        "allocated": 1,                              # Quantity used by the selected plan.
        "remaining": 1,
        "utilization": 0.5,                          # allocated / capacity.
        "saturated": false                           # True when no usable capacity remains.
      }
    ]
  },

  "diagnostics": {                                  # Summary counts and output-truncation indicators.
    "principal_unstruck": 0,                         # Number of principal targets not selected for a strike.
    "principal_unstruck_target_ids": [],
    "assets_without_feasible_strikes": 0,            # Assets whose types had no feasible strike combinations.
    "assets_without_feasible_strike_ids": [],

    "targets_total": 1,
    "targets_struck": 1,
    "targets_unstruck": 0,

    "locations_truncated": false,                    # True means the underlying collection exceeded its output cap.
    "candidate_edges_truncated": false,
    "threat_edges_truncated": false,
    "chosen_moves_truncated": false,
    "feasible_strikes_truncated": false,
    "feasible_isr_truncated": false,
    "asset_steps_truncated": false,
    "target_outcomes_truncated": false,
    "strike_events_truncated": false,
    "observation_events_truncated": false,
    "skipped_approved_targets_truncated": false,
    "platforms_truncated": false,
    "selected_loadouts_truncated": false,
    "effectors_truncated": false
  }
}

```

Maestro parses the JSON, updates the operation's solve record (setting it to completed and storing the parsed result), and separately deletes/reconstructs the group-visible Track/Engage rows and assignments. It stores the result on the Postgres workflow-status copy read by Unified.

Two outputs now exist: The solve result JSON returned by the model, and the Execute Board data (Track/Engage rows and assignments) that Strato can query.
- These writes are separate transactions. A successful solve-record update does not guarantee that the board projection or copied workflow-status update also succeeded. 

In Execute... (`/unified/execute`)

So how does Maestro rebuild the Execute Board/Strato Board/Track/Engage rows?
- First, Maestro marks the operation's solve row as completed and stores the parsed model result. 
- Next it deletes and rebuilds the group's Engage rows and asset assignments.
- It may then optionally, for one hardcoded workflow, load scenario seed data)
- It then separately updates the workflow-status copy that Unified reads.
- ==Note==: Each step can commit before the next begins; if the solve-row transaction succeeds and the board-rebuild transaction fails, the model result is durable, but Strato still contains the previous board. Unified can therefore report a completed solve while Strato shows data from an earlier publication.
- ==Note==: Temporal retries the entire storage activity after a failure. Because the callback does not carry a Temporal run ID, activity attempts, or publication version, Maestro cannot reliably distinguish a retry of the same publication from a different workflow run publishing newer data.
![[Pasted image 20260827145958.png]]

==Note==: While Temporal knows the workflow run, the current activity, every attempt, and the final close state, Unified doesn't query that history directly; it asks Maestro for status, and Maestro answers from Postgres records that it maintains as a copy of Temporal state.
- The authoritative execution state is maintained by Temporal, which knows the workflow ID, run ID, current activity, attempts, cancellation, and final close state.
- Maestro then copies it via a background status copier, an in-memory watcher that follows workflows started by that Maestro process. The copies are written to `temopral_workflow` in Postgres.
- Unified then queries (via Maestro) these Postgres copies, combining the projected workflow status with a separate `atlas_wizard_solve` row's status/result, and derives labels such as `pending`, `processing`, `completed`, `failed`, `active`, or `expired`.
	- ==Note==: These two records are updated by different code paths and can temporarily (or, after a partial failure, persistently) disagree. It's possible to have a "completed" solve beside a "pending" workflow, for instance, even though both labels refer to the same operation.


So the model result somehow converts model results into the Track and Engage rows that Strato displays.

For each matched target, Maestro creates an Engage row containing the proposed time on target, target-quality value, effectors, and attack-asset assignments.
- Targets without that generated attack-tasking row remain in the Track stage.

These rows are stored under the workflow group, they don't retain operation, temporal workflow ID, or temporal run ID. 
- When another solve for the same group is published, Maestro first deletes and then rebuilds the group's current view. 
- ==Note:== Therefore Strato is a mutable board, not a versioned archive of each source.

An operator can move a target from Track to Engage. This action changes Postgres rows...

Back in C2E land, now...

C2E doesn't have one record or ID that follows the request end to end:
Unified creates an operation ID for the user’s request. Maestro derives a Temporal workflow ID from it, and Temporal creates a separate run ID for each actual execution. The later artifact, result, and Strato records retain only some of those identifiers.
![[Pasted image 20260827191827.png]]
An investigator can't use one correlation key to prove which Temporal run produced a particular artifact.


_________

`/unified/c2e`:  Where an operator requests and reviews a model-generated plan for one operation.
- Using approved CONOP and operation identified in the URL, it loads the approved COA workflow, approved targets and their ranks, target reports, approved platform/munition combinations, current geometry snapshots, the selected operation and any previous solve result. 
- The operator then selects what to solve. The operator selects targets and may select a munition and platform for each one. These selections are then assembled into a request containing approximately:
```
{
  workflowId,
  workflowGroupId,
  operationId,
  operationName,
  operationStartDate,

  approvedTargets: [
    {
      targetId,
      rank,
      formation,
      supportedFsts,
      selectedMunition,
      selectedPlatform
    }
  ],

  waypoints,
  airControlMeasures,
  groundControlMeasures,
  maritimeControlMeasures,
  targetAcquisitionAreas,
  unitBoundaries,
  restrictedAirspace,
  refuelingPoints
}
```
- Clicking generate then starts the orchestration: Maestro starts a Temporal orchestration workflow, the orchestration assembles model inputs, the solver runs, and progress/failure information is reported to the C2E page.
- The complete JSON result is sent to Maestro to store. This result indicates much more than just a target-to-weapon sasignment, including:
	- Target schedules and times and target
	- Effector selections and counts
	- ISR custody windows
	- Realized track-quality (TQ) values over time
	- TQ gates
	- Asset travel routes
	- Asset roles and taskings
	- Threat exposure information
	- Objective breakdowns
	- Solver statistics
- The result is then rendered; the C2E page reads the saved result and transform it into the Weaponeering Timeline, which displays:
	- One group per scheduled target
	- A selected effector
	- Time on target
	- ISR custody intervals
	- Shooter movement
	- Track-quality progression
	- TQ gates
	- Assessment activity
- The map also receives plan-derived schedule and travel information.
	- ==Note==: Does this actually happen? My Map doesn't seem to update when we generate an ISR/Strike tasking.
- ==Note==: The C2E page also checks Execute assignments. There's one secondary dependency in the other direction: C2E queries the Execute-stage rows to determine whether targets already have platform and weapon assignments. It uses that information to populate displayed munition/platform values, mark a target as fully assigned, prevent already-assigned targets from being submitted again.
	- ((Might warrant more looking...))
- How does it depend on the model result?
	- It depends directly on the complete stored result. The C2E renderer consumes substantial parts of:
		- `plan_response`: `target_schedules`, `realized_tqs`, `effector_tq_gates`, `asset_travel_logs`
		- `extended_results`: `asset_taskings`
	- This result remains associated with its operation/solve, so C2E can retrieve the result for that specific planning attempt. 


`/unified/execute`: Mutable group-wide board representing the current execution state.
- The C2E page loads current group state, querying for:
	- Red assets: Observed targets and their current kinematics/confidence
	- Blue assets: Available ISR and attack platforms
	- Find rows 
	- Fix rows
	- Track rows 
	- Engage rows
	- Assess rows
		- (==NOTE:== In the current Atlas implementation, only Track and Engage are meaningfully wired into the C2E/Execute data path.)
	- Asset and weapon assignments
	- Execute events and alerts
	- ISR readiness
- This produces the five tables displayed in the left panel (Find, Fix, Track, Engage Assess)
- Track displays targets without an active engagement record.
	- Track row points to a red asset, a target report, and the workflow group. Most displayed information (position, confidence, target number, rank) comes from those referenced records.
	- The operator can assign/unassign ISR/blue assets, change required track confidence, select ENGAGE, select DROP (==Note:== the current Drop implementation doesn't appear to persist the deletion.)
- Engage displays targets with an engagement plan
	- An engage row refers to the same target and target report, but adds planning fields such as "X-time, Offset from X-time, current TQ snapshot, required TQ snapshot"
	- Separate assignment rows identify the weapon and assigned attack/ISR platforms.
	- The operator can:
		- Assign or unassign assets.
		- Abort the engagement, returning the target to Track.
	- The page also generates alerts when current confidence or TQ falls below required levels.
- Execute changes are database mutations.
	- Unlike in the Weaponeering Timeline, Execute is an interactive state. Manually engage a track target:
		- Creates an Engage row
		- Transfers existing assignments
		- Adds the selected weapon and attacker
		- Deletes the Track row
		- Reloads the group's current stage state
	- Aborting does the reverse.
- How does it depend on the model result?
	- Depends indirectly on a small projection extracted from the results.
	- When a solve completes, Maestro performs approximately this translation:
		- `target_schedules[].target_number`: Finds the corresponding red asset and target report.
		- `target_schedules[].tot`: Produces Engage X-time and target offset.
		- `target_schedules[].effector_info`: Creates the weapon assignment and required-TQ value
		- `realized_tqs`: Supplies a current-TQ snapshot
		- `extended_results.asset_taskings` with role `atk`: Creates attack-platform assignments
	- For every target schedule the Maestro can match:
		1. It creates an Engage row
		2. It deletes a matching Track row
		3. It creates weapon and attack-platform assignemnts.
	- It doesn't copy the entire result into Execute. In particular, Execute doesn't preserve the complete:
		- Custody schedule
		- Travel logs
		- TQ history
		- Objective breakdown
		- Solver statistics
		- Threat-cost analysis
	- Those instead remain in the complete result used by C2E.

==Important considerations:==
- Track rows are not generally created from the solver result... they're initially created from approved target reports. The solve only replaces successfully matched Track rows with Engage rows.
- Execute rows don't retain a clear reference to the operation or solve that produced them, they're scoped by workflow group.
- Publishing a later solve deletes and rebuilds the group's Engage rows. It can therefore replace timing and assignments produced by an earlier solve, or modified by an operator.
- Execute can function without a model result. Operators can work with seeded or live assets, assign ISR platforms, and manually transition targets between Track and Engage.
	- Therefore the model result is one producer of Execute state, not Execute's complete source of truth.
- ==Summary==: C2E retains and visualizes the complete result for one solve, while Execute receives a lossy, mutable projection of selected schedule and tasking fields for the workflow group.

```
1. Concept-fire plan is approved, at the end of P2C
   ↓
2. Workflow is selected as the approved Atlas CONOP, back at the main page
   ↓
3. Maestro creates Track rows for its approved target reports
   ↓
4. Operator runs C2E
   ↓
5. Targets scheduled by the solver become Engage rows
```
Re #3, specifically: When `setAtlasApprovedConop` runs, Maestro:
1. Deletes the workflow group's existing Track rows.
2. Loads the target reports belonging to the workflow's approved concept-fire plan.
3. Excludes areas already represented by Engage rows.
4. Creates one Track row for every remaining target report.
The intended meaning is:
> Approving the CONOP places its targets into the Execute tracking queue (via Track rows, likely) before any C2E solve occurs.

So Suppose that we have something like:
```
Approved target reports:

T-002
T-055-105
T-903A-906

Execute state immediately after CONOP selection:

Track
├── T-002
├── T-055-105
└── T-903A-906

Engage
└── empty
```
And then the C2E process results in scheduling of attacks against only `T-002` and `T-055-105`:
```
Execute state after result publication:

Track
└── T-903A-906

Engage
├── T-002
└── T-055-105
```
So the solve-publication code deletes any matching ==Track== rows to prevent those from appearing in both Track and Engage.

The implementation state machine is effectively something like:
```
Approved CONOP
      ↓
    Track
      ⇄
   Engage
   
   
NOT 
Find → Fix → Track → Engage → Assess
```

Aside, on "target reports"
- We mean Maestro's `atlas_target_reports` table
- The name is somewhat misleading; it behaves more like a *prioritized target-list entry for one concept-fire plan*.
- It might look something like this:
```json
{
  id: "target-report-uuid",

  workflowId: "workflow-uuid",
  conceptFireId: "concept-fire-uuid",

  assetId: "red-asset-uuid-for-T-055-105",

  rank: 2,
  targetNumber: null,
  targetLabel: "BE1002",

  supportedFsts: [17, 23],

  assignedEffector: "unassigned",
  assignedAttacker: "unassigned",

  createdAt: "2026-05-15T08:00:00Z",
  updatedAt: "2026-05-15T08:00:00Z"
}
```
Following `assetId` to the red-asset table might produce:
```js
{
  platformInstanceId: "T-055-105",
  platformType: "CG",
  formation: "Shandong CVBG",
  location: [6.8355361, 109.1335156]
}
```
Together, these records mean:
> In this concept-fire plan, the Chinese cruiser `T-055-105` is the second-priority target, is displayed as `BE1002`, and supports FST records 17 and 23.

Where do the target records come from? The Atlas target-extraction pipeline creates these rows before the C2E solve.
The sequence is:
```
Concept-fire plan and FSTs
            ↓
Target-extraction pipeline
            ↓
Prioritized target list
            ↓
Match each extracted target to an Atlas red asset
            ↓
Create atlas_target_reports rows
            ↓
Use those rows as approved C2E targets
```
For each extracted target, Maestro attempts to match the extracted target identifier to an existing `atlas_red_assets` row. If it cannot find a matching red asset, it drops that extracted asset, instead of creating an Atlas target report. In Atlas mode, these rows are generated automatically; the manual target-report creation mutation explicitly rejects Atlas workflows.

Q: Why do both red asset and target reports exist?
A: They store different physical information. **Red Asset** represents the observed physical entity (formation, position, altitude/speed/heading/etc. Values can change as new observations arrive) and the **target report** represents the planning decision about that entity (which workflow selected it, which concept of fires, it priority, its target label, which FSTs it supports, initial attacker/effect assignment strings. These are specific to a planning context.) 
The same red asset could therefore have different target reports in different workflows or concept-of-fires plans:
```
Red asset: T-055-105
│
├── Target report for Concept Fire A
│   ├── rank: 2
│   └── supported FSTs: [17, 23]
│
└── Target report for Concept Fire B
    ├── rank: 5
    └── supported FSTs: [31]
```
Track and Engage rows reference both records:
```js
{
  redAssetId: "physical-entity-record",
  targetReportId: "planning-decision-record"
}
```
This allows Execute to obtain live target state from the red asset, and rank/label/FST relationships from the target report.



Aside on Track Rows, Engage Rows
```
Workflow group: A contaienr that can hold one or more workflows. The execute page is loaded by workflowGroupId, not by a particular C2E or operation or solver run.
│
├── COA workflow: A specific COA-planning record
│   ├── targets and target reports
│   ├── candidate concept-fire plans
│   └── approved concept-fire plan
│
├── C2E operation / solver run: The particular C2E planning/solver run you see in `/unified/c2e`, with its own operationId, status, selected targets, solver result.
│   └── complete solver-result JSON
│
└── Execute state
    ├── Track rows
    ├── Engage rows
    ├── asset and weapon assignments
    └── other Find/Fix/Assess rows
```

A track row contains:
```json
{
  id: "34b0…",                   // UUID of this Track-stage record
  createdAt: "2026-05-15T…",
  updatedAt: "2026-05-15T…",

  workflowGroupId: "8f91…",      // Which Execute board/scenario owns it
  redAssetId: "aa12…",           // Reference to the observed physical entity
  targetReportId: "bb23…"        // Reference to the targeting decision
}
```
An engage row contains:
```json
{
  id: "91cf…",                    // UUID of this Engage-stage record
  createdAt: "2026-05-15T…",
  updatedAt: "2026-05-15T…",

  workflowGroupId: "8f91…",
  redAssetId: "aa12…",            // Same physical target
  targetReportId: "bb23…",        // Same targeting decision

  xTime: "2026-05-15T09:00:00Z",  // Common reference time for the plan
  offsetFromXTime: 420,            // This target's TOT is X-time + 420 seconds
  currentTq: 8,                    // Latest realized TQ returned by the solve
  requiredTq: 12                   // TQ required by the selected weapon
}
```

---

P2C GEOMETRIES
- ==Unit Boundaries (UBs)==, Polygons: Operational-area boundaries. You select one boundary as the principal/reference boundary, which scopes forces and planning geometry.
- ==Waypoints (WPs)==, Points: Air movement and launch locations. They become nodes in the air threat map and possible air strike launch sites.
- ==Air Control Measures (ACMs)==, Polygons: Air operating/orbit areas. Current code primarily uses each feature's centroid as an air routing and ISR sensor location.
- ==Maritime Control Measures (MCMs)==, Points: Maritime movement, ISR, and strike locations. Maritime platforms can act from these sites.
- ==Ground Control Measures (GCMs)==, Points: Ground movement, ISR, and strike locations. Ground platforms can act from these sites.
- ==Target Acquisition Indicators (TAIs)==, Polygons: Areas associated with RED targets. Targets are assigned to TAIs, and their centroids are used for strike range and sensor-effectiveness calculations.
- ==Restricted Airspace (RAS)==, Polygons:  Optional exclusion areas. Air graph edges intersecting these polygons are removed.
- ==Air-to-Air Refueling Points (AARs)==, Points: Optional air-network nodes. Today they're added to the air TWM graph, alongside WPs and ACMs. Their existence doesn't imply that we have a detailed fuel/refueling model, we don't.

Maestro converts the uploaded source



--------------

References


Ref alkemaf234: 
> There is also a likely defect in that dropdown path. At the configured Aegis commit, the helper returns raw WpnNet rows containing keys such as `Threat` and `Munition` ([athena_client.py (line 125)](/Users/sam/code/smackspace/services/aegis/app/tools/athena_client.py:125)), but the consuming activity reads lowercase `target`, `munition`, and `attacker` from those rows ([blue_attack_options_activity.py (line 26)](/Users/sam/code/smackspace/services/aegis/app/temporal/activities/blue_attack_options_activity.py:26)). Those fields are absent, so the activity should discard every row and return an empty option list. That is a static-code conclusion; we do not have a production response capture to confirm whether another worker version is actually running.


_______________________

Notes from John's Plane Wiki posts

The three services relevant to this vertical: `sisrs`, `aegis`, `wizard`. 
This page is going to document the `wizard` service, which is a pipeline that executes hen a planning request is made today.

Routing
- `maestro` is the Temporal client that starts the weaponeering/optimization workflows.
- Its routing switch (`wizard-orchestration-workflow.ts`) reads a `USE_WIZARD_SERVICE` config flag; when true, it starts workflow `wizard_model_orchestration_workflow` on task queue `wizard-model-queue`.  This deployment's `maestro` configuration has this flag set to `True`, so this is what should be happening.
	- Otherwise, it starts the subtly different `wizard_orchestration_workflow` on task queue `wizard-queue`, a workflow embedded in Aegis.

 
Wizard's WizardOrchestrationWorkflow.run() executes Five Temporal acitivties in this order:
1. TWM `wizard_twm_activity`
2. ETL `wizard_etl_activity`
3. Precompute `wizard_legacy_solve_activity`
4. Solve `wizard_store_solve_activity`
5. Store Solve `wizard_store_solve_activity`

All five are registered in a single worker in in `app/temporal/worker_registry.py`

```
WORKER_REGISTRY = {
    "wizard-model-orchestration-worker": {
        "workflows": [WizardOrchestrationWorkflow],
        "activities": [
            wizard_twm_activity, wizard_etl_activity, wizard_precompute_activity,
            wizard_legacy_solve_activity, wizard_store_solve_activity,
        ],
        "task_queue": "wizard-model-queue",
        "max_concurrent_activities": 1,
    },
}
```
This `max_concurrent_activities: 1` serializes execution. One worker process runs one pipeline instance at a time.


