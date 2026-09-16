

![[Pasted image 20260831120310.png]]

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
	- ==Target Area of Interest (TAIs)==, Polygons: Areas associated with RED targets. Targets are assigned to TAIs, and their centroids are used for strike range and sensor-effectiveness calculations.
- ==Restricted Airspace (RAS)==, Polygons:  Optional exclusion areas. Air graph edges intersecting these polygons are removed.
- ==Air-to-Air Refueling Points (AARs)==, Points: Optional air-network nodes. Today they're added to the air TWM graph, alongside WPs and ACMs. Their existence doesn't imply that we have a detailed fuel/refueling model, we don't.

Maestro converts the uploaded source



--------------

References


Ref alkemaf234: 
> There is also a likely defect in that dropdown path. At the configured Aegis commit, the helper returns raw WpnNet rows containing keys such as `Threat` and `Munition` ([athena_client.py (line 125)](/Users/sam/code/smackspace/services/aegis/app/tools/athena_client.py:125)), but the consuming activity reads lowercase `target`, `munition`, and `attacker` from those rows ([blue_attack_options_activity.py (line 26)](/Users/sam/code/smackspace/services/aegis/app/temporal/activities/blue_attack_options_activity.py:26)). Those fields are absent, so the activity should discard every row and return an empty option list. That is a static-code conclusion; we do not have a production response capture to confirm whether another worker version is actually running.


_______________________

# Plane Wiki Notes

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
- ==Note==: There isn't really a documented rationale for choosing 1, AFAICT. Likely it's a resource-safety limit for memory and CPU-intensive model preparation and optimization.
	- If request A is solving (Stage 4), request B's TWM activity (Stage 1) cannot begin on that worker, even if it's a very lightweight request.
	- The Helm defaults configure on Wizard replica with autoscaling disabled; This gives the deployment only one activity slot total, so a long-running solve can block TWM/ETL/Precompute/Solve/Storage for every other request.


Artifact Handoff between stages:
- TWM, ETL, and Precompute each write their output to Iris object storage and pass a small reference struct, `ArtifactRef`, containing `{uri, bucket, key, content_type, size-bytes, sha256, created_at_utc}`.
- The Solve stage returns its data as a plain Temporal activity result dict ,and is passed directly into the Store-Solve's activity input, without offloading to an object storage artifact.
- ==Note==: Store-Solve receives both the result and the entire `WizardWorkflowParams` object. Those parameters include the large geometry strings. Wizard also copies that parameter object into every preceding activity nput. Consequently, the workflow is offloading the intermediate outputs but still repeatedly storing potentially large original inputs in Temporal history.

> [!NOTE]- Q: When should I offload returned data from an activity to object storage, vs actually returning the main data itself?
> An activity return doesn't get automatically passed from one worker invocation to the next; Temporal serializs and durably records it in the workflow history; if the workflow then supplies that value to another activity, Temporal serializes it again as the next activity's input. This makes large inline handoffs both a size-limit risk and a history-growth cost.
> The design principle is: **Temporal should carry control data, while object storage should carry bulk data.**
> In In Wizard, this is only partially applied; TWM/ETL/Precompute return references because their outputs contain large tables, maps, and matrices. The solve result is returned inline because it was apparently assumed to be smaller.

> [!NOTE]- Q: How does activity data handoff actually work?
>  If Activity A returns `result`, and the workflow passes it to Activity B:
>  - Activity A's worker serializes `result` and sends those bytes to the Temporal SErvice
>  - Temporal stores them in the workflow history's `ActivityTaskCompleted` event.
>  - The workflow worker loads that history and decodes `result`.
>  - The workflow schedules Activity B with `result` in B's input.
>  - Temporal serializes and stores those byte again in the `ActivityTaskScheduled` event.
>  - The worker running B retrieves and decodes that input.
>So there's no direct memory transfer from Activity A to Activity B.
>```
>Activity A worker
 >   ↓ result bytes
>Temporal history: ActivityTaskCompleted
 >   ↓ workflow replay
>Temporal history: ActivityTaskScheduled(input=result)
 >   ↓
> Activity B worker
>```
>So passing a 300KB result from A to B generally adds 600KB to workflow history, plus event and wrapper overhead.
>Regarding Temporal's limits, the current standard limits are:
>- 2 MiB for an individual payload/event blob
>- 4 MiB for a gRPC message or Event History transaction
>- 50 MiB or 51,2000 events for one Workflow Execution.
>So TLDR:
>Use a normal activity return when the value is: 
>- Small and has a known upper bound
>- Needed for workflow decisions
>- An identifier, status, count, timestamp, compact summary, or short collection.
>- Useful when inspecting Temporal history.
>- Not expensive to serialize repeatedly.
>
>==Note:== The current solve result being passed inline is probably only acceptable if Wizard establishes and enforces a bounded serialized size, and currently it does not. Feels like we should probably just use the same artifact pattern for the solve output, along with (as mentioned earlier in this document) artifact the large initial geometry/input bundle once and pass its reference to all five activities, which makes Temporal history consistently contain identifiers, references, statuses, and summaries, not repeated model datasets.

```
Wizard activity worker
    │ reports Solve result
    ▼
Temporal Server
    │ stores ActivityTaskCompleted event
    │ creates a workflow task
    ▼
Wizard workflow worker
    │ replays WizardOrchestrationWorkflow
    │ reaches the completed Solve await
    │ schedules Store-Solve
    ▼
Temporal Server
    │ stores ActivityTaskScheduled event
    │ creates an activity task
    ▼
Wizard activity worker
    executes Store-Solve
```
Note that "workflow worker" and "activity worker" are conceptual roles; in this deployment, they're both handled by the same Wizard SDK `Worker` instance and usually the same service process. The SDK machinery that polls/replays is internal to the Temporal Python SDK, but the worker itself is operationally ours, running in the Wizard container, consuming Wizard resources, and stopping processing if the Wizard service is unavailable.

Every stage that needs KG data (TWM, ETL, Precompute) opens its own `AthenaKnowledgeGraphClient` (Wizard's `app/tools/graph_lib.py`), which is a gRPC client to the separate `athena` Go service; there's no direct Cypher/Neo4j driver usage in this repo.
- ==Note==: The client's constructor accepts `uri`, `user`, and `password`, but only uses `workflow_group_id`; the connection is `grpc.insecure_channel(settings.ATHENA_URL)`.
- Each stage opens its own client independently rather than sharing one.
- ==Note:== TWM, ETL, and Precompute each call `get_wpnNet()` separately (for the weapon-engagement reasons), so the same weapon-network data is fetched from Athena three times per pipeline run.
# (1) [Threat Weighted Map (TWM) Activity](https://work.smackgov.com/product/projects/5582eab5-bd31-46cb-8f2b-77ccd8333b8b/pages/ec47cc0d-5b00-4d0a-823c-b1dbb9e000d8/)
- The first stage of the `wizard` pipeline, builds, for every combination of movement mode and BLUE platform type, a threat-weighted adjacency matrix over the relevant control-measure nodes (waypoints, ACMs, GCMs, MCMs), scoring how dangerous each edge in the routing graph is, based on RED formation's proximity and weapon range.
- What it does:
	1. Parses GeoJSON control measures from the request (waypoints, ACM/GCM/MCM, TAI, unit boundaries, AAR, RAS, plus the principal boundary polygon).
	2. Opens an AthenaKnowledgeGraphClient and pulls:
		1. `get_wpnNet`
		2. `get_red_instances`
		3. `get_blue-instances_in_range`
	3. For every (mode, blue_platform_type) combination, build a `ThreatWEightedMap`
		- Note: "Mode" means the operating domain of a BLUE platform (1:Air, 2:Ground, 3:Maritime). It selects which control-measure network the platform is assumed to move through.
			- Note that Wizard also defines (4:Subsurface, 5:Space), but the TWM implementation ignores them.
		- Node set by mode: Air uses Waypoints (WP) + ACM (+AAR in theory) (Strike from WP, ISR from ACM), Ground uses GCM, Maritime uses MCM
		- Edge pruning: An edge is dropped if it crosses a RAS polygon, or if it's a long haul (>300km direct distance) without a valid intermediate node within 50km perpendicular of the direct line.
		- Edge Weight: Sums, over every RED instance with weapon range of the edge, `1.0 - threat_modifier * (dist_km/max_range)`, clamped to `[0,1]`.
		- Stated again, differently:
			- It builds a separate, domain-specific complete graph for (effectively) each BLUE platform type (e.g. F-35A, MQ-9).
			- It does this by first grouping the BLUE platform types in the scenario by mode (Air, Ground, Maritime), and then builds a separate TWM for each *platform type.*
			- Say we have :
			-  ```
			  BLUE assets

			F-35A_01    platform_type=F-35A    mode=AIR
			F-35A_02    platform_type=F-35A    mode=AIR
			MQ-9_01     platform_type=MQ-9     mode=AIR
			DDG-51_01   platform_type=DDG-51   mode=MARITIME
			  ```
			- Wizard doesn't build four TWMs for this, it deduplicates this into: `(AIR, F-35A), (AIR, MQ-9), (MARITIME, DDG-51)`. Why aren't F-35 and MQ-9 combined? They actually start with the same air graph, but assume there's a RED SAM nearby; Athena weaponNet might tell Wizard that its max range against F-35A is 40km, while against MQ-9 it's 100km. So for a given edge, the RED SAM might contribute 0 threat against the F-35, and .4 threat against the MQ-9. So you have the same physical graph for the F-35 and MQ-9, but a different threat-cost matrix. Different F-35s might have different starting locations, weapons, assignments, etc, but the question "How dangerous is it for an F-35 to traverse WP-A -> WP-B" only needs to be answered once, here.
				- ==Note:== We're making some simplifying assumptions here, largely around the computing of the TWM only being dependent on the BLUE-effecting RED platform's weapon characteristics... and not on (e.g.) the specific BLUE instance's loadout (do you have flares? chaff? external weapons increasing RCS?).
- Output is per-(mode, platform_type) matrices; each populated cell is (score, contributions), plus a summary (platforms, total_edges, red_instances_count, etc.)
- Writes the result to the shared blob store.
	- ==Note==: `normalize_twm_for_etl` seems to be an identity no-op right now?
- I/O:
	- Input: `TWPParams`: GEoJSON control measures, boundary polygon, workflow group id
	- Output: 
```json
{
	"platforms": {
		"nodes": [...],
		"node_coords": {...},
		"matrix": [[null, [0.42, {"RED-UNIT-1": 0.42}], ...], ...]
	},
	"summary": {
		"total_platforms": 6,
		"total_edges": 340,
		"red_instances_count": 12
	}
}
```
Later:
- ETL reads the TWM artifact back, to validate that its shape exists. The payload's content is not otherwise read by ETL.
	- Note: ==That's weird.==
- Precompute consumes TWM's content via `extract_targetable_costs_from_twms(all_twms)`, which flattens the per-platform threat matrices into a flat `{from_node}-{to_node}->{platform_type: score}` dict feeding the the solver's targetable-cost terms.

So what again exactly is a TWM?
- It's useful to think of *one TWM* as approximately:  *"A lookup table saying how threatening each candidate movement segment is to a particular BLUE platform type.*
Conceptually:
```
# One TWM
ThreatWeightedMap {
    platform_type: "F-35A"
    mode: AIR

    edges: {
        WP-A ↔ WP-B:   0.00
        WP-A ↔ ACM-C:  0.25
        WP-B ↔ ACM-C:  0.10
        ...
    }
}

# And another TWM
ThreatWeightedMap {
    platform_type: "MQ-9"
    mode: AIR

    edges: {
        WP-A ↔ WP-B:   0.40
        WP-A ↔ ACM-C:  0.70
        WP-B ↔ ACM-C:  0.55
        ...
    }
}
```
==Except it's actually a little more detailed than the above!== 
- The model needs to distinguish **fixed** threats from **targetable** threats. If a SAM-1 is itself something that BLUE can destroy, its contribution needs to be identifiable so that the solver can effectively reason that (WP-A -> WP-B) costs +0.40 threat when alive, but after SAM-1 is destroyed, that portion of the threat cost disappears.
	- ==Fixed== doesn't actually mean stationary, it means a RED threat whose destruction is not something this optimization is allowed/able to model. This split is based on whether the RED threat represented in the TWM can also be treated as a target by the C2E solve.
		- The split is based on whether the RED threat represented in the TWM can also be treated as a target by the C2E solve.
			- Note "Can this RED thing threaten BLUE" and "Can BLUE destroy this RED thing in this solve" don't necessarily have the same answer. A RED entity can be threatening but not prosecutable.
			- Fixed = "Assume this threat remains alive for the whole plan.", Targetable="The optimizer may destroy this threat, so stop charging its route-threat contribution after its destruction."
	- ==Note:== There's sort of an implicit assumption here that we just never want to target anything outside of the set of targets from our C2E step... which... seems true at first-pass, but you can imagine that a commander would like to know if additionally targeting one additional 
	- ==Note:== It seems like with our TWM we're also making the assumption that threat is just like related to closeness to a red platform (given its range), without regard for whether that target has sensors to target you, is that right? 
- So the TWM isn't merely a static "heat map," it's threat information attached to candidate route segments. This is also why mode and platform_type do different jobs:
	- mode: Which network should this thing move on? e.g. AIR -> WP/AAR/ACM
	- platform_type: How threatening are RED weapons to this thing? F-35A gets one set of edge weights, while MQ-9 gets another set of edge weights.
	- **Takeaway**: Mode determines the shape of the TWM graph, while platform type determines the threat values painted onto that graph.-

Node types in the current TWM model:
- GP (Generation point or home/base): Mobile assets start here; connects to only one entry node.
- WP: Aircraft can move through it, hold there, and launch strikes.
- ACM: Aircraft can move through it and hold there for ISR.
- GCM: Ground assets move and act here.
- MCM: Maritime assets move and act here.
- TAI: Not a movement node; is it to be observed or struck at a standoff.
- RAS: Not a node; used to remove prohibited air edges that pass through it.
- AAR: Included in TWM generation, but currently removed before the solve, because refueling isn't modeled.

Aside: What's up with these Generation Points?
- It seems from the code that assets must start at their own base, and cannot visit another asset's GP. Aircraft must nominally return to their own base.
- A GP is a location where Wizard considers a BLUE asset to start its plan and rest when it is not operating.
- In practical terms, it is an abstraction for a home airbase, port, staging area, or the asset's current location.
- In the Knowledge-graph-reading path, Athena contains to related pieces of data:
	1. A named GP node with `type:gp` with coordinates.
	2. A BLUE asset whose `generation_point` property names that node.
```json
A named GP node with coordinates
{
	"labels": ["ElementInstance", "Infrastructure"],
	"name": "Andersen AFB",
	"type": "gp",
	"lat": 13.584,
	"lon": 144.9245
}

# And then a BLUE asset whose generation point names that node
{
	"name": "F-35A_001_USA",
	"lat": 14.1,
	"lon": 145.2,
	"generation_point": "Andersen AFB"
}
```
- Alternatively, a BLUE asset doesn't have to reference a named GP, and can instead contain "generation_point":"self", or if the generation_point property is missing, it's treated the same: A generation point is created  at the lat/lon of the unit.
```json
{
  "name": "DDG-51_001_USA",
  "lat": 25.3,
  "lon": 129.8,
  "generation_point": "self"
}
```
==Note:== This seems to me to be a brittle connection; we know for instance that Nick has been renaming some "names" because they weren't display-ready. If name is basically being used as a foreign key for such things, and we changed the name of Andersen AFB, that would probably break something here. Instead, should it be something like a Neo4j relationship such as "BASED_AT"?
- ==Note==: Currently, if an asset names a GP that cannot be found, Wizard silently uses the asset's current coordinates while retaining the missing GP's name. This can conceal bad scenario data.

How is the candidate edge set built?
- Initially, ETL creates a fully connected grahp; every location is connected to every other location, plus self-edges for holding station. 
- The solve them cleans that graph, removing node classes that aren't traversable, including TAIs.
- Dividing nodes into air/ground/maritime domains
- Removing cross-domain edges
- Connecting each mobile base to one nearest own-domain entry node:
	- Air GP -> Nearest WP
	- Ground GP -> Nearest GCM
	- Maritime GP -> Nearest MCM
- Remove air edges that cross restricted airspace.
- Prune the graph to edges involved in candidate strike and ISR routes.

==Note==: An important TWM/movement mismatch:
- TWM doesn't authoritatively define the movement graph.
- Wizard builds the movement network and the threat map separately, and the two stages don't agree on which edges exist.
	- During the solve, a movement edge without a TWM record is treated as having zero threat, not as forbidden or unknown.
	- This creates routes that are legal to the solver, but were never evaluated for threats.
From beginning to end:
1. The request supplies geographic points and polygons (WP-A, WP-B, WP-C, RA polygons, BLUE aircraft and their GPs from Athena, RED SAM systems from Athena)
2. TWM independently decides which edges to evaluate: For an F-35, TWM creates an air-node set containing waypoints, ACMs, and AAR. It examines possible edges from pairs of these waypoints, and drops those that are longer than 300km that have a valid intermediate node within 50km perpendicular of the direct line, with the assumption that the aircraft should use the two shorter legs.
So if we have:
```
WP-A ─────── 200 km ─────── WP-B ─────── 200 km ─────── WP-C

We examine possible pairs:
WP-A ↔ WP-B
WP-B ↔ WP-C
WP-A ↔ WP-C

TWM then produces something like:
WP-A ↔ WP-B: threat score 0.8
WP-B ↔ WP-C: threat score 0.3
WP-A ↔ WP-C: no TWM entry, since the edge is >300km and WP-B is close to the stright line between them.
```
this "No TWM entry" doesn't mean that the threat score is zero, it means that TWM deliberately didn't evaluate that edge.
TWM also drops air edges if it crosses restricted airspace.

Later, ETL separately reconstructs the movement graph
- ETL doesn't start with TWM's edge list; it independently creates a fully-connected graph:
```
WP-A ↔ WP-B
WP-B ↔ WP-C
WP-A ↔ WP-C
```
- Every location is initially connected to every other location, where by "location," we're referring to all of (WP, ACM, GCM, MCM, TAI, AAR, UB, RAS, GP). The principal boundary itself is not added as a location, nor are RED unit positions. A BLUE's current position becomes a location only if Wizard uses it as a "self" GP (see further above in the document). 
- ETL then creates an edge for every unordered pair of rows in the table, so the raw graph can contain:
```
WP-1  ↔ WP-2
WP-1  ↔ ACM-1
WP-1  ↔ GCM-1
WP-1  ↔ MCM-1
WP-1  ↔ TAI-1
WP-1  ↔ RAS-1 centroid
ACM-1 ↔ GCM-1
GCM-1 ↔ MCM-1
GP-1  ↔ every other location
```
- Before the graph rereaches the solver, it is substantially rewritten:
	- GCM/AAR/TAI/UB/RAS locations are removed
	- WP/ACM/MCM remain
	- Original GP edges are removed
	- Colocated GPs are consolidated
	- Each air GP gets one edge to its nearest WP
	- Each maritime GP gets one edge to its nearest MCM
- The resulting production movement nodes are therefore approximately:
```
Air:       GP → WP and movement among WP/ACM nodes
Maritime:  GP → MCM and movement among MCM nodes
```
((And note that Eli has some other version of Wizard that allows for ground movement for a NEC demo, but that isn't in Main currently.))
==Note:== that production revision retains original edges between any surviving interior nodes... So a cross-domain edge such as a `WP->MCM` and an `ACM->MCM` can also survive. 
- TWM never creates these cross-domain edges, so they don't have a corresponding TWM threat evaluation.

> ETL initially creates a direct edge between every pair of retained control-measure centroids and BLUE generation points, inclduing WP, ACM, GCM, WCM, TAI, AAR, UB, RAS, GP rows. Solve preprocessing later removes non-operational node classes and rebuilds GP connections.

Okay, back to the list...

3. ETL then separately constructs the movement graph
- ETL doesn't start with TWM's edge list; it independently creates a fully-connected graph:
```
WP-A ↔ WP-B
WP-B ↔ WP-C
WP-A ↔ WP-C
```
- Every location is initially connected directly to every other location.
- Later processing removes out-of-domain nodes and reduces the graph, but it doesn't use teh TWM matrix as the authoritative list of allowd edge.s
- Consequently, the previous 400-km-direct-edge can remain available to the solver ,even though the TWM removed it.


4. GP edges are added only to the movement graph.
- Suppose the F-35 starts at `gp_air_1`. Wizard adds: `gp_air_1 <-> WP-A`, but GPs do not exist in the TWM node set, therefore TWM cannot calculate a score for this edge.
- So now we have something like this:

| Edge              | Movement graph | TWM                  |
| ----------------- | -------------- | -------------------- |
| `gp_air_1 ↔ WP-A` | Yes            | Never evaluated      |
| `WP-A ↔ WP-B`     | Yes            | Score `0.8`          |
| `WP-B ↔ WP-C`     | Yes            | Score `0.3`          |
| `WP-A ↔ WP-C`     | Yes            | Deliberately omitted |

5. The solver loads the movement and threat data:
- It treats the movement garph as the authority for where the aircraft can travel.
- Separately, it converts TWM contributions itno a threat-cost table keyed by `movement edge + BLUE platform type`, e.g.:
```
WP-A-WP-B + F-35A → 0.8
WP-B-WP-C + F-35A → 0.3

But there is no threat row for:
gp_air_1-WP-A + F-35A            # GPs don't exist in the TWM node set, so TWM didnt' calculate a score for this edge
WP-A-WP-C + F-35A                # The TWM activity removed this, because >300km and there were nodes within ~50km perpendicular of the edge
```

6. Missing threat rows become zero cost
==Note:== This is clearly wrong 😄
- When Wizard joins threat costs onto the aircraft's possible movements, it uses a left join and fills missing fixed costs with zero:
	- `move_cost_fixed = move_cost_fixed.fillna(0)` ==Bad! Should there ever even be NAs?==

7. The optimizer may then select the unevaluated route (which actually *will* in reality have non-zero threat)
The optimizer compares:
```
WP-A → WP-B → WP-C
Threat cost: 0.8 + 0.3 = 1.1

# Against

WP-A → WP-C
Threat cost: 0.0
```
It might choose the direct route, because it appears threat-free, but TWM never concluded that the direct route was safe; it never calculate the route at all! 
- This is a **core error:** "No threat data exists for this movement edge" is interpreted as "this movement edge has zero threat."

Restricted airspace makes this even more serious:
- TWM currently removes air edges that cross RAS, but the primary movement-graph builder doesn't apply the same restriction.
So we can get:
```
Movement graph: allowed
TWM: omitted because it crosses restricted airspace
Solver threat cost: zero
```

So it seem like there ==should== be one authoritative edge list:
1. Build every legal movement edge once.
2. Apply domain and RAS restrictions there.
3. Add GP entry edges there.
4. Calculate TWM costs for every resulting edge and platform type.
5. Reject the payload (failing loudly) if an allowed edge does not have an explicit threat result.

The threat result *should* extinguish "This edge was explicitly evaluated and has a score of 0" with "Not evaluated"
Today, those two cases collapse into the same zero-cost behavior.


# (2) [ETL Activity](https://work.smackgov.com/product/projects/5582eab5-bd31-46cb-8f2b-77ccd8333b8b/pages/a0dfe4fb-b927-4d7b-8473-d4be13ccb65c/)'
- The second stage of the `wizard` pipeline, which builds the full set of dataframes that the solver needs: laydown, movement graphs, engagement-capability joins, distance tables, and feasible-strike combinations, from the knowledge graph, or from static CSV overrides when supplied.

What it does:
- Reads back the TWM artifact to validate its shape exists ((and then... doesn't use it again))
- Re-parses the same GeoJSON control measures and boundary used in TWM
- Branches on whether static laydown override were supplied (params.laydown_blue/laydown_red, CSV strings)
- Builds movement/edge graphs, waypoint/acm/tai location tables, engagement capability join tables, a full cross-join distance table, and a geometric feasibility join of loadouts x Pk x range x standoff constraints (an asset's effect must be able to reach a TAI from a waypoint within range, respecting inflight standoff)
- Optionally subsamples assets/targets/edges for smoke testing
- All 21 dataframes are serialized to JSON strings per-column

The output shape has 22 required keys (`ETL_REQURIED_KEYS`):
```
df_gps, df_wps, df_acms, df_tais, df_locs, df_moves, df_edges, df_ec, df_ec_tt,
df_loadouts, df_laydown_red, df_laydown_blue, df_pk, df_pstats_blue, df_pstats_red,
df_estats_blue, df_isr_capes, df_tq2tle, df_dep_options, df_sat_obs, df_loc2loc,
df_feasible_strikes, num_assets, num_targets, num_control_measures
```
Each df_* field is a JSON-serialized string, deserialized downstream in Precompute, Solve)
- Precompute deserializes `df_laydown_blue`, `df_isr_capes`, `df_laydown_red`, and `df_pstats_blue` directly from this artifact.
- Solve deserializes the full `ETLResult(**etl_payload)` dataclass.


# (3) [Precompute](https://work.smackgov.com/product/projects/5582eab5-bd31-46cb-8f2b-77ccd8333b8b/pages/6a503c5a-8756-4685-8cb4-472165c3c574/)
- Turns the ETL and TWM outputs into the three data structures that the solver's objective/constraints use:
	- Sesnsor Efficacy (Target Location Error)
	- Weapon Effectiveness (per-shot Pk)
	- Targetable Costs (flattened threat scores)
What it does:
- Reads back both the TWM and ETL artifacts
- Deserializes `dflaydown_blue`, `df_isr_capes`, `df_laydown_red`, `df_pstates_blue` from the ETL artifact's JSON strings.
- Opens a fresh `AthenaKnowledgeGraphClient` to fetch `wpn_lookup`
- Sensor efficacy:
	- Launches two Temporal child workflows on task_queue="sensor-detection-queue", run concurrently via asyncio.gather: server_side_batch_workflow (for non-satellite sensor coverage) and server_side_batch_satellite_prepare_workflow (==satellite== pass geometry)
	- Computes Target Location Error (TLE, in km) per (platform_type, sensor_modality, ACM, TAI, target_type) tuple, plus satellite-only rows (df_sats_obs) separately.
- Weapon effectiveness:
	- Converts a cumulative p_k into a per-shot individual kill probability p_i.
- Targetable costs: 
	- Flattens TWM's per-platform threat matrices into a flat dict keyed as `{from_node}-{to_node} --> {platform_type:score}`
- `twm_data` in the output is the entire TWM payload reserialized alongside the newly computed sensor/weapon/cost tables.
	- ==Note:== Seems weird to reserialize a previous step's payload like that, rather than passing a reference?

Output shape (dictated by `PRE_COMPUTE_REQUIRED_KEYS`):
```
twm_data, sensor_efficacy, weapon_effectiveness, targetable_costs,
num_platforms, num_edges, num_red_instances, satellite_observations
```

Downstream consumption:
- The Solve activity deserializes the full `PrecomputeResult(**precompute_payload)` dataclass alongside ETLResult, feeding both into the Gurobi/GenEx pipeline.


# (4) [Solve](https://work.smackgov.com/product/projects/5582eab5-bd31-46cb-8f2b-77ccd8333b8b/pages/8bebd459-7416-4cf1-8e9d-524fb758bac5/)
- The fourth stage of the `wizard` pipeline. Builds a Gurboi MIP model from the ETL and Precompute outputs, applies an approved target list, and dispatches the solve (typically to a remote Gurobi service over HTTP)
- Reads back all three artifacts (TWM, ETL, Precompute)
- (The John Notes aren't useful for this, it's LM slop that assumes knowledge)


# (5) [Store Solve](https://work.smackgov.com/product/projects/5582eab5-bd31-46cb-8f2b-77ccd8333b8b/pages/e30c0484-fec0-4d76-aa92-74d606005ea2/)
- Fifth stage of the `wizard` pipeline; persists the solve rseult to Maestro via gRPC
- `normalize_solve_result`  unwraps the raw solve activity's dict into the final exported shape, handling several possible result  shapes: an `exported_resul` key, or a model+params+policy+target_list bundle, or raw dfs+lp
	- ==Note:== Whats up with the several possible result shapes?
- `store_solve_payload` opens a gRPC channel to Maestro, and stores the solve; the gRPC call is how the solve result reaches Maestro's data store.

Input/Output: The Solve stage's raw output dict, plus `conop_id`, `operation_id`, `temporal_workflow_id`
Output: None

__________

Using my file:///Users/sam/code/smackspace/docs/investigations/unified-c2e/activity-deep-dive.html document, which is supposed to be a step-by-step walkthrough of all operations.

# Activity 1: Threat Weighted Map (TWM)
- Creates one graph per distinct in-range BLUE platform type.
	- On the graph, a node is a permitted movement location.
	- An edge is a direct route between two nodes.
	- An edge score is the sum of proximity-based contributions from RED weapons that can engage that BLUE platform type.

> [!NOTE]- Q: What does "In Range" mean, re: "distinct in-range BLUE platform types"?
> It means that a unit is geographically close-enough to the principal operating boundary, based on the BLUE platform type's own maximum weapon range.
> The code does this:
> 	1. Finds the largest Range value in WpnNet where the BLUE platform type is the attacker.
> 	2. Expands the principal-boundary polygon outwards by that distance.
> 	3. Includes a BLUE platform instance if its current latitude/longitude falls inside the expanded polygon.
> 	4. Creates a TWN graph for each distinct qualifying platform type, not for each instance.
> So if an F-35 type's WpnNet entries have range of 370km and 925km, an F-35 instance qualifies if it is inside the principal boundary or within 925km of its exterior.
> If the platform type has no matching WpnNet entry, the expansion distance defaults to zero, so the instance must be inside the original boundary. 
> So "in-range' doesn't mean that any particular *target* (or any RED unit) is within range, nor that a platform's selected munition can reach a selected target.
> Note that the inclusion distance (polygon "expansion" distance) is platform-specific; an F-35's longest weapon doesn't expand the eligibility area for a B-52 or a destroyer. Each type gets its own buffer.
> ==Note:== The code takes the largest range from any WpnNet row for a particular platform type, but doesn't check whether that weapon is relevant/compatbiel to a nominated target, and definitely doesn't check whether that weapon could reach compatible targets, just that the weapon can reach the box. Note that if a particular platform type has a 10,000 km weapon, BLUE instances of that type could qualify from thousands of kilometers away; it wouldn't expand eligibility for every particular BLUE type, but it could make that particular type effectively global. 

> [!NOTE]- Q: What about how we include RED platforms?
> Even RED instances that are not nominated as targets contribute to the TWM, and even RED instances that aren't within the principal boundary can effect TWM edge weights.
> For each RED instance, TWM does this independently:
> 	1. Look up the RED platform type's longest-range weapon in WpnNet
> 	2. Expand the principal boundary by that range
> 	3. Include the RED instance if its current location is inside that expanded area.
> Then, when we construct a graph for a particular BLUE platform type, an included RED instance affects an edge ONLY IF
> - WpnNet say s that the RED platform and one of its munitions can attack that BLUE platform type.
> - The *edge* passes within that specific munition's range of the RED instance.
> So a RED instance can be initially included because its weapon can reach into the operating area (principal boundary) with any weapon, but then not actually effect an edge either because the edge is outside range, or because the RED platform can't effect our BLUE platform with any of its munitions.
> Note that both targeted and non-targeted ("fixed") RED units can contribute to edge weights, not only the RED entities selected as targets in the operation.


Inputs read:
- WP, AAR, ACM, GCM, MCM, TAI, UB, and RAS GeoJSON strings
- Principal-boundary GeoJSON
- `workflow_group_id`
- Athena WpnNet engagements
- Athena RED and BLUE element instances

Request fields NOT read:
- Approved targets, including selected platform/munition
- BLUE or RED laydown overrides

Output:
- `platforms`: One node list and symmetric cost matrix per platform type
- `summary`: Platform, edge, mode, control-measure, and RED-instance counts
- Iris `ArtifactRef` with bucket/key/byte count/digest/timestamp


Steps:
1. Reduce the workflow request to only TWM-specific fields
	- We construct a new TWMParams object containing geometry strings and the workflow-group ID. This is not a pass-through of the complete request; approved targets, laydown overrides, and model settings are excluded.
	- Consequence: An operator-approved target outside ordinary TWM RED-instance range is not force-retained merely because it was approved.
2. Parse each geometry category independently
	- Each input string must decode to a GeoJSON `FeatureCollection` or `Feature`
	- A feature name comes from `properties.name`; absent names become `unnamed_N`. Duplicate names within one category are silently skipped, and features without geometries are also skipped.
	- All geometry types are reduced to a Shapely geometry plus its centroid. Later, node calculations only use the centroid coordinates.
	- Problem: A polygon's full shape matters for RAS intersection and TAI membership, but most location tables and graph nodes preserve only the centroid. Also, duplicate and (more important) missing features can disappear without a validation error (unsure if there's logging).
3. Parse one principal-boundary polygon
	- If the boundary is a FeatureCollection, only its first feature is used.
	- The resulting geometry must be a Polygon; a MultiPolygon is rejected. This polygon becomes the spatial reference for filtering BLUE assets, RED threats, and selected control-measure categories.
	- Problem: The assumption that the first feature is the complete authoritative operational boundary, and that additional features in the boundary collection are ignored without warning.
4. Open an Athena client scoped by workflow group
	- ==Note==: The constructor accepts Neo4j/URI/username/password, but the implementation uses ATHENA_URL and an insecure gRPC channel.
	- ==Note:== Instance, generation-point, and sensor queries carry the workflow-group context, while WpnNet and platform-stat queries do not carry that context in their request types.'
		- Specifically, this means whether the gRPC request includes the "workflow_group_id" key under the "context" key in the sent JSON. Athena can then restrict its Neo4j query to scenario nodes belonging to that group, specifically, nodes marked as scenario data whose `athena_workflow_group_id` equals the supplied ID. This prevents one C2E group from reading another's. The Assumption being made here is that group context is used for the laydown/instance graph, while global WpnNet rows are appropriate for this run.
			- Element-instance and generation-point queries include the ID, and Athena uses it to filter Neo4j results.
			- WpnNet, platform-stat, and effector-stat request types contain only fields such as team. Those queries read the shared type-level catalog, not a group-specific laydown.
			- ==Note:== Sensor-capability requests include the context field, but the inspected Athena implementation currently ignores it and reads the shared sensor catalog.
		- If no workflow-group ID is provided, the scoped Neo4j select base, non-scenario nodes rather than nodes from every workflow group.
5. Read the complete weapon network and index it twice
	- Each Athena engagement (A WpnNet capability record returned by Athena, which says something lke (AttackerType, Munition, TargetType, Range, Salvo Size, Capacity) becomes two dictionary entries:
		- `by_target[target]`: "Which attacker and munition can threaten this target platform type?" Identifies RED platform types and weapons that threaten the BLUE platforms.
		- `by_attacker[attacker`: "Which target types can this attacker engage, with which munition?" Identifies RED platform types that can be effected by BLUE platforms with some munitions.
	- Both carry range, salvo size, capacity, alert rate, magazine depth, and speed.
6. Read and spatially filter RED instances
	- Athena returns RED instances only when lat/long/underscore in a name are present. For each RED platform type, Wizard finds that type's maximum weapon range in WpnNet and buffers the principal boundary by that distance; the instance survives if its point is inside the corresponding buffered polygon. 
	- Each surviving red instance will receive a deduplicated list of positive-range effectors from `by_attacker[platform_type]` (What BLUE instances can this RED instance engage, and with what munitions?)
	- Note: Difference from ETL: TWM doesn't provide an `include_instance_ids`; ETL later force-retains approved target IDs; TWM does not.
7. Read and spatially filter BLUE instances
	- Athena returns BLUE instances with coordinates, underscore-bearing names, platform metadata, mode, and generation-point name.
	- Wizard reconstructs the platform type from the first and last underscore-separated name segments.  So `F-35B_001.VMFA-122.MAG-13.3MAW_USA` becomes platform type `F-35B_USA`.
	- It then retains only air, ground, or maritime modes and only instances inside the principal boundary buffered by that platform type's maximum weapon range.
8. Collapse BLUE instances to distinct platform types by mode
	- TWM doesn't build a graph per instance, it builds a graph per distinct platform type, so all F-35B instance share one `F-35B_USA` TWM, which will tell about their aggregate risk from all relevant RED platforms/munitions). 
9. Filter control measures using a 230km approximate buffer
	- The boundary buffer converts kilometers to degrees using a 111km/latitude degree and a cosine-adjusted longitude estimate, then averages those values into one degree radius.
		- WP, TAI, and MCM records are exempt and always retained, but AAR/ACM/GCM/UB/RAS records survive only when their centroid is inside the buffered polygon.
	- This is a coarse geographic filter intended to discard map features that are too far from the operation area.
		- Wizard first enlarges the principal boundary by a 230km margin, and then decides which control measures to retain.
		- Currently, we keep every WP, TIA, MCM record, regardless of location (even if they're thousands of kilometers away).
		- For AAR/ACM/GCM/UB/RAS, we keep only if its center point is inside the principal boundary, plus the 230 km margin.
		- The kilometer-to-degree part is necessary, because the GeoJSON coordinates are long/lat, measured in degrees, while the desired margin is specified in kilometers. The code approximates 1 degree of latitude for 111 km, and 1 degree of longitude to 111 * cos(latitude) km. It averages these two values and uses the result as one buffer distance in degrees. Wizard then enlarges the polygon by approximately that many degrees.
			- ==Note:== Is this a good assumption to make? Is there a better way to do it? Probably create an accurate 230km buffer using a suitable projected coordinate system or geodesic distance calculation, not an approximate conversion from kilometers to degrees. We could then include geometries according to their shape (point, line, polygon might have different rules, though I think we're all using polygons for these). The current centroid test can exclude a large polygon that significantly overlaps the relevant area just because its center is outside, or it can include a large polygon whose center is inside, even though almost all of it is outside (maybe fine), and it can treat equivalent distances differently depending on latitude.
			- That is not a true 230 km geographic buffer. Longitude and latitude degrees represent different physical distances, and the difference grows at higher latitudes. Averaging them into one number produces an uneven real-world margin.
10. Select Graph Nodes by Platform Domain
	- Air graphs use WP + AAR+ ACM centroids. Ground graphs use GCM centroids. Maritime graphs use MCM centroids.
		- TAI polygons, unit boundaries, principal boundaries, RAS polygons, and generation points are not TWM nodes.
		- This is because the graph represents candidate travel locations; TAIs are target referents, and RAS polygons are route-exclusion geometry.
	- ==Note:== ETL initially makes every location category (including TAI and GP) part of one complete graph; TWM, in contrast, uses separate domain graphs with a smaller node vocabulary.
11. Generate candidate direct edges and remove two classes
	- Wizard evaluates every unordered graph node pair (A+B, A+C, etc.)
	- For air graphs, it drops an edge when its straight line intersects any parsed RAS polygon. 
		- ==Note:== Interestingly, intersection ues the original unfiltered RAS list, not the boundary-filtered list.
	- For every mode, a route longer than 300km is dropped when any same-domain node lies within 50km of the straight segment.
		- Important: 300km isn't a maximum edge length... a 600km edge still remains if no intermediate node lies within 50km of its segment.
12. Calculate each RED-weapon contribution to an edge
	- For the current BLUE platform type, Wizard finds RED platform types that can engage it.
	- For each matching RED instance and each listed effector, it computes the approximate distance from the RED point to the straight route segment. If the distance is within weapon range, the contribution is 1 - distance / range, clamped to 0-1.
		- A RED weapon with 100km range located 20km from the route contributes 1-20/100 = 0.8
	- The final edge score is the sum of all contributions from all red relevant red platforms. It's not clamped, so two threats can produce a score greater than one.
		- ==Note:== Interesting that the single-threat contribution is clamped, but not the summed contributions.
13. Store a symmetric matrix for each platform type:
	- Each retained edge is written to both `matrix[i][j]` and `matrix[j]i]` (meaning A->B and B->A cost the same to transit). 
	- Each cell contains:
		- `score`
		- A contribution map keyed as `<red instance name>_<effector name>`. Diagonal and omitted edges are null.
	- ==Note:== Wizard doesn't just choose *one* RED munition, it adds every qualifying munition to the edge threat value.
		- So if `SAM_007_CHN` has 3 effectors (HQ-9, HQ-16, MANPAD) that can effect an F-35, each of them would have a contribution (0.7+0.6+omtited beecause out of range of edge = 1.3)
		- The matrix cell would look like:
		- ```
			{
			  "score": 1.4,
			  "contributions": {
			    "SAM_007_CHN_HQ-9": 0.8,
			    "SAM_007_CHN_HQ-16": 0.6
			  }
			}
		  ```
		- So there's no decision like "The SAM would probably use HQ-9;" both weapons are treated as indepednent, simultaneous sources of route threat, and their values are added together.
		- ==Note:== That means a platform with more compatible weapon types can generate a larger threat score merely because more weapons are listed in WpnNet. The calculation does not account for which weapon is loaded, which weapon would actually be used, ammunition quantity, or mutual exclusivity between weapons.
	- ==Note:== I'm also realizing that there's nothing here about the actual effectiveness of enemy munitions. If you're at the tail end of range away from a highly deadly munition, versus close to a not very dangerous weapon, the not very dangerous weapon has a larger contribution. There's nothing about the weapon's characteristics (or requirements in terms of RED tq) incorporated into the threat value.
	- ==Note:== TWM itself doesn't decide whether a RED threat is fixed or targetable; that classification is introduced later, and joined back to the TWM data during Solve. TWM itself writes matrix cells like the code block above: For a given route edge, when used by a friendly platform, we the aggregate score, and the contributions to that score from various red platform/effector pairs. The later ETL activity assigns a preliminary classification when it builds a RED laydown table, using ("RED point inside a TAI polygon = Targetable, vs Red point outside every TAI = fixed). If a RED point is outside every TAI, ETL associates it with the nearest TAI, but still labels it `fixed`.
		- ==Note:== This is crazy, lol. So it's the Geometries we upload (which include TAIs) that determine if a RED unit is targetable, rather than the FST target list from the P2C flow, or the checkbox selection of targets in the weaponeering step of the C2E flow?
			- Those checkbox-selected targets are called 'Principal' targets in the code, it seems. RED units can be principal/not principal and inside TAI/outside TAI.
			- ETL force-retains them even if they are outside the normal RED geographic filter
			- Solve designates surviving, strikeable selected targets as "principal"
			- The objective gives large reward for destroying principal targets.
			- ... Anyways, some weird contradictory behavior exists:
				- A selected principal target outside every TAI can be destroyed for the principal target reward, but its threat contribution remains a fixed route cost after its destruction.
				- An unselected RED unit inside a TAI is treated as a targetable ***residual*** threat. The solver may decide to attack it solely to remove its contribution from a route.
				- So: "In the configured Wizard revision, TAI geometry—not the operator’s selected target list—determines the preliminary `fixed` versus `targetable` classification used for route-threat costs. The selected target IDs are handled separately: they are force-retained during ETL and later designated as principal targets during Solve. Consequently, a selected principal target outside every TAI can still have `fixed` threat costs, while an unselected RED unit inside a TAI can be treated as a targetable residual and attacked to reduce route cost." The world "targetable" is badly overloaded here: It does not mean "selectable by the operator as a target," it means "the model may remove this units route-threat cost after a modeled strike."
			- If "targetable" just means "blowing it up would remove its effect from the TWM," shouldn't this apply to every red unit? I don't see the point of only including those that are in TAIs.
14. Convert platform failures into data, validate only the top-level keys, and write Iris
	- We store exceptions while building a platform map is caught and stored as `{"error": "..."}`, and the activity can still succeed.
		- So `failed_platforms > 0` doesn't fail the stage. Precompute later SKIPS errored platform maps.
			- ==Note:== Per-platform errors can occur only when constructing that platform's map. In contrast, invalid top-level GeoJSON, an Athena failure, failure to parse the principal boundary, or failure to load WpnNet will all fail the entire TWM activity, normally.
				- Per-platform errors could be: 
					- Invalid or malformed coordinates passed to Shapely.
					- A geometry error while intersecting an edge with a RAS polygon.
					- A nonnumeric or otherwise malformed weapon value used during threat calculation.
					- An unexpected missing field or incompatible type.
					- Memory exhaustion while creating a very large N × N matrix.
					- An implementation bug triggered by a particular platform or domain.
				- So this is a pretty shit pattern that probably shouldn't be doing what it's doing.
				- Suppose we had:
				- ```
					{
					  "platforms": {
					    "F-35B_USA": {
					      "error": "some geometry failure"
					    },
					    "MQ-9A_USA": {
					      "nodes": ["WP-1", "WP-2"],
					      "matrix": [...]
					    }
					  },
					  "summary": {
					    "total_platforms": 2,
					    "successful_platforms": 1,
					    "failed_platforms": 1
					  }
					}
				  ```
				  -  TWM will still succeed, because Validation checks only that the top-level platforms and summary keys exist; it doesn't reject failed_platforms=1
				  - ETL will then succeed independently (it reads the TWM artifact and performs the same top-level validation for some reason, but does not use its platform matrices)
				  - Precompute skips the failed map in one derived calculation; When Precompute creates its confusingly-named `targetable_costs` object, it does this: `if "error" in twm_data: continue`. Consequently, it produces no edge-score entries for the F-35! It still copies the original TWM payload into its output artifact.
				  - The configured solve then probably fails. It rereads the raw TWM payload and iterates every platform entry. It assumes each entry has `nodes` and `matrix` keys, so we'll get a KeyError and fail the solve activity. Retries will encounter the same problem.
				  - Recap: "A per-platform exception does not fail TWM. The error is embedded in the artifact, and top-level validation accepts it. ETL does not use the platform matrices and continues. Precompute omits the failed platform when deriving aggregate edge scores but preserves the original error entry. The configured Solve loader does not handle that entry and expects `nodes` and `matrix`, so the failure is deferred until Solve rather than safely degraded."
Example TWM data:
```json
{
  "platforms": {
    "F-35B_USA": {
      "platform_type": "F-35B_USA",
      "mode": 1,
      "mode_name": "Air",
      "nodes": ["Cy 11", "ACM23"],
      "node_coords": {
        "Cy 11": [121.02, 20.81],
        "ACM23": [121.55, 21.10]
      },
      "num_edges": 1,
      "matrix": [
        [null, {"score": 0.8, "contributions": {"SAM_007_CHN_HQ-9": 0.8}}],
        [{"score": 0.8, "contributions": {"SAM_007_CHN_HQ-9": 0.8}}, null]
      ]
    },
    "MQ-9_USA": ... # Adding this just to show that there could be multiple platforms
  },
  "summary": {
    "total_platforms": 1,
    "successful_platforms": 1,
    "failed_platforms": 0,
    "total_edges": 1,
    "platforms_by_mode": {
      "Air": 1,
      "Ground": 0,
      "Maritime": 0
    },
    "control_measure_counts": {},
    "red_instances_count": 1
  }
}
```

(Non-exhaustive problems of TWM; Still look through the above "Note" highlighted stuff to get more. Some rewrite tips are in the html doc)
- Unsafe modeling assumption: Weapon range being used as the platform-availability radius. A BLUE instance qualitfies when it's near enough to the boundary area according to the longest weapon ever associated with its platform type. Doesn't establish that the weapon is carreid in teh current loadout, or that the wepaon can affect any approved or present target.  There's no maximum cap, so an intercontinental-range entry can make a paltform type qualify from an operationally meaningless distance.
- TWM and ETL can disagree about which RED instances exist
	- TWM retains RED instances by a per-type weapon-range buffer
	- ETL applies the same filter but additionally force-retains approved target IDs
	- An approved target can therefore appear in the optimizer's red laydown without having contributed any threat data to the TWM. 
- Platform types and threat contributions depend on parsing names
	- BLUE platform type is reconstructed from underscore-separated instance names instead of using the Athena-returned platform-type field?
		- Not sure what hte best is here
- Valid geometries can be ignored without stopping the run
	- E.g. only the first principal-boundary feature is used, duplicate control-measure names are skipped, several spatial tests use centroids, and kilometer buffers are approximated in longitude/latitude degrees. These shortcuts can execute a polygon that intersects ht operating area or produce large errors for very long range and high latitudes.
- The threat score has no defensible physical interpretation. 
	- The score is a linear function of perpendicular distance to a strait segment. Contributions are summed without a maximum, then copied symmetrically to both travel directions. This is neither a probability nor an expected loss measure, but we treat it like an objective cost.
- A partially-failed map can be published as a successful activity.
	- A per-platform exception becomes an `error` object inside the artifact. Precompute skips that platform, while top-level validation still passes.



# Activity 2: Extract, Transform, Load
- ETL converts geometry and Athena records into the dataframe schemas expected by the legacy optimizer. It's output is an outer JSON object containing 22 dataframe values, each itself encoded as a JSON string, pluss three counts.

Inputs:
- All geometry and the workflow-group ID
- Approved target IDs
- Optional BLUe/RED laydown JSON
- Debug sampling/filter controls
- Athena instances, generation points, WpnNet, platform/effector stats, and sensor labels.

Output:
- 22 schema-shaped dataframe strings
- `num_assets, num_targets, num_control_measures`
- Iris artifact reference


1. TWM Check
	- ETL downloads the referenced TWM object and performs the two-top-level-key validation that TWM does too. It then calls `run_wizard_etl(input.params)` with only the original request parameters no TWM instances enter ETL.
		- ==Note:== This is weird that ETL does this repeated check and then doesn't do anything with ETL, right?
2. Reparse and refilter the original geometry
	- ETL repeats the GEoJSON parsing, principal-boundary selection, and 230km filtering performed by TWM. It doesn't reuse TWM's parsed objects or record the filtered set as a shared run snapshot.
		- ==Note:== The GeoJSON parsing twice between activities (which require serialization/deserialization at edges) isn't a problem, it's the filtering decision being made twice on geometries (the 230km exclusion policy.)
	- Each surviving control measure becomes a location row in `df_locs` with `loc_id`, `loc_tt`, and `(lat, lon)` centroid. Here, `loc_tt` would be values like `wp`, `acm`, `gcm`, `mcm`, etc.
		- Aside: `tt` = "Type", e.g. effector_tt is weapon type, loc_tt is location type, etc.
	- ```
	  loc_id   loc_tt   loc_ll
	ACM23    acm      (21.10, 121.55)
	`Cy 11    wp       (20.81, 121.02)
	TAI-4    tai      (20.95, 121.70)
	  ```
	  - Generation points are subsequently added to this same table, using `loc_tt = "gp"`
3. Choose overrides: one of the two source-system branches
	- If both laydown overrides are null, ETL uses Athena/KG data, but if either override is non-null, ETL switches to the alternate branch and uses the provided JSON for each supplied side. It BLUE is missing, it uses the bundled BLU laydown, while if Red is missing, it synthesizes RED target rows from TAI locations.
	- Weapon, platform, and effector tables come from bundled CSV data, and Athena remains in use only for sensor labels.
		- The two override parameters are `laydown_blue` and `laydown_red`; Under the normal product path, BOTH SHOULD BE NULL.
		- The unusual alternate-data path activates if either parameter is non-null. 
4. In the normal branch, reread WpnNet and graph entities
	- ETL opens a new Athena client and independently reads WpnNet, generation-point nodes, RED intsancse, BLUE intsances, platform statistics, effector statistics, and sensor capabilities.
		- ==Note:== Note that a graph change between last read of WpnNet and this read of WpnNet can alter platform types, weapon ranges, target inclusion, GP coordinates, or sensor eligiblity. We should probably be snapshotting or something.
5. Force approved target IDs into the ETL RED set
	- ETL extracts only `target_id` from each approved-target record and passes those IDs to RED-instance filtering... a matching ID survives even if it falls outside the platform-specific buffered boundary.
		- The idea is that an operator-approved target should remain available to the model, even if ordinary range scoping would omit it.
			- ==Note:== This forced inclusion is absent from the TWM, so an approved target can exist in t`df_laydown_red` without ever contributing to a TWN edge.
				- Q: Wait, isn't there a guarantee that these RED platforms not included in TWM would also not influence TWM edges, since it's a stricter condition?
				  A: Not necessarily. That conclusion would be valid only if every TWM edge were guaranteed to lie inside the principal boundary. The implementation does not guarantee that.
				- Recap: "ETL force-retains approved RED target IDs, but TWM does not. An approved target outside TWM’s principal-boundary-based RED filter can therefore appear in df_laydown_red without any TWM contribution. That absence cannot be interpreted as confirmed zero threat because TWM can contain route nodes and edges outside the principal boundary, and the two activities read and transform their inputs independently. If every route edge were entirely inside the principal boundary, and both calculations used the same distance model (they don't; The boundary buffer and point-to-edge calculation use different geographic approximations) and WpnNet snapshot, then a RED instance outside the boundary by more than its longest weapon range could not threaten any such edge."
	- Rank, formation, supported FSTs, selected munition, and selected platform are not used.
6. Classify each RED instance by TAI membership
	- ETL tests each RED point against TAI polygons. The first containing TAI assigns `tai_id` and `cost_tt="targetable"`; if the point is outside every TAI, ETL still assigns the nearest TAI, but leaves `cost_tt="fixed`".
		- `targetable` and `fixed` describe whether the threat instance is inside a target area... and those that are inside target areas mean that their contribution to edges is removed after the target's modeled ==destruction time.==
			- Destroying a `targetable` target effectively disables its TWM edge contributions for later route traversals, but note that the stored TWM matrix is never directly modified or recalculated. It just determines how Solve interprets an existing contribution over time. A targetable contribution can stop being charged after the target's modeled destruction time; a fixed contribution remains charged for every traversal. The TWM artifact itself remains unchanged.
7. Build BLUE and RED laydown rows
	- Each BLUE instance becomes `(asset_id, asset_tt, asset_domain, gp_id`)
		- ```
		asset_id:     F-35B_001.VMFA-122.MAG-13.3MAW_USA
		asset_tt:     F-35B_USA
		asset_domain: air
		gp_id:        gp_F-35B_001.VMFA-122.MAG-13.3MAW_USA
		  ```
		- Platform type are uppercased in the BLUE table 🤷
	- Each RED instance becomes (`target_id`, `target_tt`, `tai_id`, `cost_tt`)
		- ```
		target_id         target_tt   tai_id   cost_tt
		T-055_105_CHN     T-055       TAI 4    targetable
		SAM_007_CHN       SAM         TAI 4    fixed
		  ```
	- These establish stable instance IDs plus type-level information to join to statistics and capabilities.
	- ==Note:== A missing `mode` becomes an air asset, rather than a validation failure. That's probably not what we want.
8. Resolve each BLUE asset's generation point. (NOTE: This actually happens like.... inside the loop of the previous step, where we obviously have the gp_id. We determine teh gp_id reference, and the resolve each gp_id to coordinates, create the transient GP-coordinate rows, append them to df_locs, and later build df_gps.)
	- If the generation point is "self", ETL creates `gp_<asset_instance>` at the asset's current coordinates, otherwise it uses the named GP node and its coordinates.
	- ==Note:== Suppose Athena returns a BLUE instance with a named generation point of `Andersen_AFB`; Wizard them looks for a generation-point record named `Andersen_AFB`. If it finds one, the result is correct, but if it doesn't find one, Wizard does not raise an error, it keeps the name but substitutes the asset's current coordinates. Downstream data therefore looks like a successfully-resolved named generation point, even though Andersen_AFB was never found. This can also have the result of result units that both airplanes refering to the same missing gp_id to silently then have the SAME fallback coordinates; It's NOT the case that they'll have different fallback locations (their own locations), thankfully, but when they're added to df_locs, Wizard deduplicates... and only the first row wins. Both aircraft are associated with the same coordinate, which came from only the first aircraft.
		- Comment: It's like at every opportunity, when we had the opportunity to loudly fail (correctly) when we have bad data, we instead decide to just paper over it silently with a shitty solution that will produce (in the worst case) wrong answers.
		- "Wizard does not distinguish “named GP successfully resolved” from “named GP missing, asset coordinates substituted.” If multiple assets produce conflicting coordinates for the same GP name, Wizard silently keeps the first coordinate and discards the others."
9. Build platform, effector, loadout, and kill-probability tables
	- ETL filters BLUE platform stats to types present in the laydown, deduplicates effectors by name, builds loadouts from `by_attacker`, and derives per-shot success from WpnNet salvo size using a hard-coded cumulative `p_k=0.9`. 
		- When we say "builds loadouts," what do we mean?
	- This step creates five type-level lookup tables.
	- Assume we had: 
	- ```
	  BLUE instances:
		F-35B_001_USA → type F-35B_USA
		F-35B_002_USA → type F-35B_USA
		MQ-9A_004_USA → type MQ-9A_USA
		
		RED instances:
		T-055_105_CHN → type T-055
		SAM_007_CHN   → type SAM
	  ```
	  - ETL needs to answer:
		  - How fast and far can each BLUE platform travel? `df_pstats_blue`
		  - What are the characteristics of each BLUE weapon? `df_estats_blue`
		  - Which weapon does each BLUE platform type carry, and how many? `df_loadouts`
			  - ```
				      asset_tt             loadout_id effector_tt  asset_effector_capacity_ct
				0    F-35B_USA  loadout_F-35B_USA     AGM-158B-2                           2
				1    F-35B_USA  loadout_F-35B_USA       AIM-120D                           4
			    ```
			- Q: So is this loadout_id col just like... bullshit? Is it just loadout_{asset_tt}? 
				- ==Note: ==A: Yeah. The solver is designed to support alternative loadout configurations (e.g. loadout_ids of F35_STRIKE, F35_AIR_TO_AIR, F35_SEAD, with different effectors and capacities, and the solver could then choose exactly one configuration for each aircraft), and weapons sharing a `loadout_id` would describe one complete, internally-compatible configuration. But in reality, ETL doesn't have those alternative configurations; instead, it takes every WpnNet weapon associated with a platform type and places them all under one synthetic loadout.
				- The solver requires every asset to choose exactly one loadout; because ETL provides only one, the choice is forced.
					- LM: "Therefore, if WpnNet’s Capacity means “maximum number when configured to carry this weapon,” ETL is interpreting it incorrectly as “number simultaneously available alongside every other compatible weapon.” In the current implementation, we "Group every WpnNet weapon associated with a BLUE platform type under one generated identifier and treat all listed per-weapon capacities as simultaneously available to every instance of that platform type." That is not a real loadout, it's a platform-weapon compatibility catalog converted into a capacity bundle.  "ETL creates one synthetic loadout per BLUE platform type. It places every WpnNet-compatible weapon under that loadout and copies WpnNet Capacity into a per-weapon capacity column. Because no alternative loadouts are generated, every asset of the type is forced to receive the same bundle. Capacities for different weapon types are treated as simultaneously available; no shared carriage, hardpoint, weight, or configuration constraint limits the combined load."
		  - How effective is each weapon against each RED platform type? `df_pk`
			  - ==Note:== WpnNet does not provide the per-shot probability used here; ETL assumes that a complete salvo has a 90% probability of success, so it then back-calculates a per-shot probability that would produce 90% cumulative success after `n` independent shots. `p_per_shot = 1 - (1 - 0.9)^(1/n)`
			  - ==Note:== Solve doesn't treat this ETL table as authoritative. ==Precompute repeats the calculation using a hard-coded cumulative probability of 0.2==
				  - ((WTF))
				  - Solve discards ETL's `df_pk` rows and rebuilds the table from the Precompute value.
		  - What are the movement characteristics of each RED platform type?: `df_pstats_red`: RED platform statistics
			  - ==Note:== If Athena has no matching record, ETL just manufactures as fallback of 
				  - ```
				    target_domain:    surface
					target_category:  fixed
					target_spd_kmph:  0
					target_rng_km:    0
				    ```
			  - ==Note:== "If RED platform statistics are missing, pretend that type is a stationary, zero-range surface target." ((Why D:))
10. Expand sensor-capable platform types to four modalities
	- Athena is queried for BLUE platform types carrying Radar, ELINT, or EOIR labels.
	- Country suffixes are normalized for matching. 
11. Test
	- Test
12. Test
	- Test
13. Test
	- Test




==NOTE:== I realized that the analysis so far was based around the commit that Chat saw in production (in some Helm chart), rather than the more current commit seen in Main, and some things have materially changed. 
==Important Changes:==
- Control-measure boundary filtering changed 
	- Old code exempted WP, TAI, and MCM from principal-boundary filtering; Main also exempts ACM and GCM; therefore all WP, TAI, MCM, ACM and GCM records are retained, regardless of their distance from the principal boundary. AAR/UB/RAS remain boundary-filtered. This can materially enlarge the graph and preserve air ISR orbits and ground launch sites that the old version discarded.
- Restricted airspace now affected Solve movement graphs: The request's RAS polygons are passed into the Solve loader; Solve removes air edges whose straight-line segments intersect restricted airspace. RAS does not filter maritime or ground edges.
- Operational domains are no longer fixed to air and maritime
	- Main determines active domains from BLUE asset types that can perform an action. It supports air, maritime, and ground. `excluded_domains` can remove domains.
		- Air: Can move through both WP and ACMs. Air ISR sites are ACMs. Air weapon release sites are WPs. AARs/TAIs are not traversable.
		- Maritime: Movement, ISR, and strikes through MCM
			- ==Note==: ISR is intended at MCM, but normal precompute does not produce MCM sensor placements
		- Ground: Movement, ISR, and strikes through GCM
			- ==Note==: ISR is intended at GCM, but normal precompute does not produce MCM sensor placements
	- ==Note:== Not sure if this is new, but one additional constraint is that each BLUE asset is assigned exactly one role for the entire solve: Attack, ISR, Escort, or Rest. 
	- ==Note:== The solve code is written to support maritime ISR from MCMSs and ground ISR from GCMs, but normal precompute only evaluates sensor performance at ACMs. The domain (solve) filter subsequently rejects ACM-based sensor placements for maritime and ground platforms because an ACM is an air-domain node...
- Cross domain movement is explicitly removed
- Static ground and maritime assets are supported
	- A speed-zero ground or maritime asset that has a sensor receives a static GCM or MCM home at its actual coordinates. It receives hold actions but no movement edges. Static air assets remain unsupported
- End-at-home behavior changed
	- Air assets finish at their home generation point, while maritime and gronud assets may finish at any reachable location in their domain.
	- Feasible strikes are rebuilt differently...
- Probability of kill authority changed
	- We resolve weapon effectiveness primarily by weapon type and target type. Platform type is retained as provenance, but longer creates separate probabilities for the same weapon-target pair.
	- Country-qualified target names take precedence during name resolution. Duplicate equivalent records collapse, conflicting probabilities for the same resolved pair cause an error.
- Approachable but unreachable targets are retained
	- Previously, downstream filtering could remove targets that had no feasible strike rows.
	- We now retain all RED targets; apn approved target without a feasible positive-PK allocation is marked strikeable-false, gets zero aggregate PK, and is reported as skipped.
	- This is important for threat modeling: An unreachable RED threat remains in the model, rather than disappearing.
- Air tasking is tied to an actual feasible strike.
	- An asset can receive attack tasking only if at least one of its feasible strike decision rows is selected. Assets with no feasible strike candidates are forced not to attack.
- Satellite selection is deterministic
	- The old path randomly sampled ten satellites and could fail when fewer than ten existed
	- Main greedily selects satellites according to coverage of distinct relevant TAI/time observations, with NORAD ID as teh time-breadker
- Target designation now happens before symetry constraints
	- Principal/residual designation and strikeability are established before target are grouped for symmetry breaking


> [!NOTE]- I really don't understand the current ISR situation, can you please explain it in more detail?
> - In this context, ISR exists specifically to support a weapon release. 
> 	1. The model considers selecting a strike
> 	2. That weapon release requires a sufficiently-accurate target observation
> 	3. That observation must be supplied by a deployable BLUE ISR platform OR a satellite pass
> 	4. If no acceptable observation exists, that strike cannot be selected
> - In Solve, every effector requires a cue observation at TQ=4, which corresponds to a maximum target-location error of approximately 21.618 km.
> - Inflight ISR updates are currently disabled, as are Terminal ISR updates.
> - So under current behavior, each selected strike requires exactly one qualifying observation before release; not "at least one"; the model constrains the observation count to exactly one.
> - The observation window is specific: With the current ten-minute cue offset, the observation must occur during the ten-minute interval that starts twentyminutes before weapon release, and ends ten minutes before release.
> - A deployable ISR platform must hold on station for at least ten minutes.
> - How BLUE platforms become ISR candidates?
> 	- ETL asks Athena for BLUE platform types associated with any of these sensor labels (Radar, ELINT, EOIR), then matches these platform types against the BLUE platform types in the laydown.
> 	- ==Note:== The capability-building code then does something questionable: Once a platform matches *any* of these sensor labels, it assigns *all four* of these modalities to the platform (EOIR, SAR, ELINT, Radar). The code does not preserve which specific source sensor label caused the platform to qualify.
> 	- Smack sensors later attempts to resolve each proposed platform/modality pair against its own sensor configuration, so not every combination necessarily survives ((?)), nevertheless, the upstream capability table overstates what the source query established. 
> - In precompute, we gather every unique RED target type in the RED laydown, every TAI in teh input geometry, every ACM in the input geometry, every candidate BLUE platform type and sensor modality. It does NOT gather MCMs or GCMs for the sensor calculation. 

==Note:== I think there's some sort of dormant logic for weapons that are configured to require an inflight update.
- `minimum launch distance = weapon speed × time from launch until the inflight update` ((Lol this seems kinda silly? It seems to mandate an inflight update?))
	- For such a weapon, a candidate must satisfy: `inflight-standoff distance < launch-to-target distance < maximum weapon range`
	- The lower bound ensures the weapon would still be airborne when the update is scheduled. For example, if the weapon travels 200 km before its scheduled update but launches only 157 km from the target, it would reach the target before the update. That candidate is rejected.
	- This only applies when inflight_tq is populated, though.
- Our payload loaders set every weapon's inflight_tq to NaN, meaning "No inflight update required," retaining only a cue-quality requirement before launch.
	- So this means that inflight updates don't affect normal solves.

> [!NOTE]- What is Cueing?
> In C2E, cueing means that before a planned weapon release, the plan must contain a qualifying sensor observation of the target's area.  A cue is a planned acquisition event.
> A cue can be provided from:
> - A deployable BLUE ISR asset if: 
> 	- Its platform type is marked `isr_taskable`
> 	- Its sensor has a precomputed observation opportunity for the relevant TAI and target type
> 	- The calculated sensor TLE meets the TQ 4 threshold
> 	- The optimizer assigns that asset to the ISR task
> 	- The asset travels to and holds at the specified observation site
> 	- Its `obs_hr` falls within the cue window
> 	- Its holding step lasts at least 10 minutes
> - An eligible satellite pass can provide the cue if:
> 	- The satellite has a precomputed pass over the target's TAI
> 	- Its fixed pass time falls within the cue window
> 	- Its predicted TLE meets the TQ 4 threshold
> With ISR enabled, selecting a strike automatically activates a cue requirement.
> Comparison with weapon release: Weapon release is a planned kinetic event, where the optimizer must choose {An attacking asset, a feasible launch location, a weapon type, a number of weapons, a release time, and a target}. The impact time (time on target) is calculated as release time + weapon travel time. The cue is what makes the corresponding strike selectable in the mathematical model.
> Note that deployable sensor efficacy is calculated for a target type at a TAI, using the TAI centroid, not for the exact nominated target's live position. 
> If we had a weapon release at 10:00
> ```
> 09:40                  09:50                    10:00 
> |---- valid cue window ----|---- assumed gap ----| weapon release
> ```
> Then you would need an observation in the 9:40-9:50 window (whether at 9:47 or 9:50) with a TLE of <=21.618km (>=TQ4)
> Note that one ISR bird at one observation time can satisfy the two requirements of two launchers that are each sending munitions at the same time. The solver represents this as two requirement-satisfication records that point to the same ISR asset, the same ISR step, and the same `obs_hr`. It doesn't create one shared "observation object." Both are satisfies if the same observation opportunity's obs_hr falls inside both cue windows. Note that in a differnt world, two launches shooting two diffetn munitions could have different TQ requirements, which may or may not be satisfied by a single observation from a single ISR platform.
> If we had two launchers both shooting at different targets in the same TAI... that same observation could still (in theory) satisfy those different cue requirements.

- Target desginatino now happens before symmetry 

______
CALL with Eli:
- The .90 and .20 p_kill stuff. Yeah, it's bad. The .90 is old shit from Andy, and the .20 was a miscommunication about how we should lower the thresholds, and instead someone lowered the probability itself. Oops!
- Loadouts: Eli recalled that these things do/should exist. He referred to the same platform, loadout_id, effector, capacity table. He also thought that they had dome some sort of mixed capacity thing where if you can carry 6 LRASMs and 2 GOOBIES, you can also carry 3 LRASMS and 2 GOOBIES, ya know. Which also isn't optimal, and should probably come from like... the knowledge graph loadouts, you know. I don't know whether we do red stuff.
- Eli said that the story was that the was doing some of this shit on his own, and Coogan strewn a bunch of Eli's research across these different activities in a temporal workflow.... and that Eli needed to be able to update it for a demo or BK or something and couldn't do it, didn't know how to run smackspace, whatever it might have been... but anyways, the point being that he had to update some stuff in the solver instead.
- I think Eli and I were talking past eachother on the TAI thing... Specifically, two things:
	- It seemed like I learned that ONLY THINGS WITHIN TAIs are targetable?
		- I thought it was the case
	- A definitional struggle on what was a principal target vs a residual target
		- Is principal + residual MECE in terms of the red units that are under consideration? It seemed like he was saying that residual targets were... non principal ones that we were still going to strike/service to clear the way?
		- It's true that principal/residual and fixed/targetable are different axes, right? It's true that fixed/targetable mean "inside TAI or not," right? So if there's an enemy SAM (principal or not) that's outside a TAI and threatening our shit... it's fixed and 

__________



# ISR-In-C2E.md file
- Target Quality (TQ): An ordinal mission-level statement of how good target information must be for an action. 
- Target Location Error (TLE): A quantitative estimate of location uncertainty, normally expressed in meters/kilometers and tied to a confidence convention. Lower TLE means a more precise estimated location.
- What changes observation usability.
- An approved BLUE laydown contains:









_______________








