---
aliases:
  - OSDK
---
[[Palantir]]'s way of turning an organization's raw data into an operational model of the real world. Described as an operational layer that sits *above* datasets, virtual tables, and models, connecting those digital assets to real-world counterparts.

> The Palantir Ontology system is Palantir’s operational representation of an organization: a shared layer of business objects, relationships, rules, actions, functions, and permissions that turns fragmented enterprise data into something humans, applications, and AI agents can reason over and act on.

Closer to a business object model plus workflow/action layer plus permission system plus AI interface.
```
Raw systems:
ERP tables, CRM records, sensor feeds, spreadsheets, ML models, APIs

Palantir Ontology:
Aircraft, Flight, Airport, Maintenance Event, Part, Supplier
Properties: tail number, departure time, risk score, inventory level
Links: Flight -> uses -> Aircraft
       Aircraft -> has -> Maintenance Event
       Part -> supplied by -> Supplier
Actions: assign aircraft, delay flight, order part, approve repair
Security: who can see, edit, approve, or trigger each thing

Applications and AI:
Dashboards, workflows, search, simulations, AIP agents, operational apps
```

|Ordinary ontology|Palantir Ontology|
|---|---|
|Defines domain concepts|Defines domain concepts|
|Defines relationships|Defines relationships|
|May support reasoning|Supports operational workflows and AI use cases|
|Often descriptive|Descriptive and action-oriented|
|Usually separates model from application|Becomes the shared layer powering applications, decisions, and agents|
Normal [[Ontology]]
```
A Flight is an event.
A Flight uses an Aircraft.
An Aircraft has Maintenance Events.
```

Palantir wants the Ontology to also support:
```
Show me all flights at risk.
Explain which aircraft caused the risk.
Recommend reassignment options.
Let an authorized operator reassign the aircraft.
Write the approved decision back to operational systems.
Log who did it and why.
```
The last part is the Palantir twist; the Onology is meant to be kinetic, not only semantic; Palantir uses "semantic elements" for objects, properties, and links, and "kinetic elements" for actions/functions.


Core Concepts:
- Object type: A type of real-world entity or event (`Employee`, `Aircraft`, `Factory`, `Order`)
- Object: One concrete instance of an object types.
- Property: A characteristic of an object, like `status`, `location`, `risk score`, or `departure time`
- Link type: A relationship between object types, like `Flight -> assigned aircraft -> Aircraft`
- Action type: A governed operation users can take, like `Assign Employee`, `Approve Order`, or `Create Maintenance Request`
- Function: Custom business logic (e.g. in Python or TS) that reads, computes over, or edits Ontology objects


# Example

We might have an airline with raw data spread across many systems
```
flight_schedule table
aircraft_inventory table
maintenance_events table
crew_assignments table
weather_feed
passenger_bookings table
```

The Palantir Ontology would model the airline's world as objects:
```
Flight
Aircraft
Airport
Crew Member
Maintenance Event
Passenger Booking
Weather Event
```

With Links:
```
Flight -> assigned to -> Aircraft
Aircraft -> has -> Maintenance Event
Flight -> departs from -> Airport
Flight -> affected by -> Weather Event
Crew Member -> assigned to -> Flight
```

With actions:
```
Reassign aircraft
Delay flight
Create maintenance work order
Notify affected passengers
Approve exception
```

Now an operational user or an AI agent can ask:
> Which flights are at risk tomorrow, why, and what can we do about them?

And the system can theoretically traverse the Ontology:
```
Weather Event -> affects -> Airport
Airport -> departure point for -> Flight
Flight -> assigned to -> Aircraft
Aircraft -> has -> Maintenance Event
Maintenance Event -> requires -> Part
Part -> supplied by -> Suppliers
```
Then the user can take a governed action, such as reassigning an aircraft, instead of merely viewing a dashboard.

The Ontology creates the translation layer between the raw data and the layer that an operator actually thinks in. [[Palantir Artificial Intelligence Platform]] (AIP) sits on top of this operational model.



____________
# From a Developer' Perspective?
- The Ontology looks less like an academic [[Ontology]] and more like a generated domain API over enterprise data.

You define or consume something like:
```ts
Flight
  properties: flightId, departureTime, status, riskScore

Aircraft
  properties: tailNumber, maintenanceStatus, location

Link type:
  Flight.assignedAircraft -> Aircraft

Action:
  reassignAircraft(flightId, aircraftId, reason)
```
Then application code talks to these concepts, instead of directly talking to raw tables.

What you actually build:
1. The model layer: Someone defines object types, properties, link types, actions, and permissions in Ontology Manager. Object types are commonly backed by Foundry datasets. A dataset might start with `flights_schedule` table and expose it as an Ontology object type `Flight` with links `Flight -> assignedAircraft -> Aircraft`.
2. The application layer: A TS/Python/Java app uses the [[Palantir Ontology|OSDK]] (Ontology Software Development Kit)... which is generated from the relevant subset of your Ontology and gives type-safe acces to objects, actions, and functions.
```ts
import { Flight, Aircraft, reassignAircraft } from "@ontology/sdk";

const delayedFlights = await client(Flight)
  .where({
    status: { $eq: "Delayed" },
    riskScore: { $gte: 0.8 },
  })
  .fetchPage({ $pageSize: 50 });

const aircraft = await client(Aircraft).fetchOne("N12345");

const result = await client(reassignAircraft).applyAction({
  flightId: "UA-2401",
  aircraftId: aircraft.$primaryKey,
  reason: "Maintenance conflict",
});
```
3. The server-side logic: For custom backend logic, you write Functions in TS or Pythons. Palantir describes Functions as a server-side isolated code that can read object properties, traverse links, and make Ontology edits.
A function computing a risk score:
```ts
export default async function calculateFlightRisk(flight: Flight): Promise<number> {
  const aircraft = await flight.assignedAircraft.fetchOne();
  const openMaintenanceEvents = await aircraft.maintenanceEvents
    .where({ status: { $eq: "Open" } })
    .fetchPage({ $pageSize: 100 });

  return openMaintenanceEvents.data.length > 0 ? 0.9 : 0.2;
}
```
This is simplified, but the developer idea is real: Function code is written against Ontology objects rather than hand-joining raw tables.
4. Actions: The most nonstandard part is that writes don't usually look like arbitrary database updates. You define action types:
```text
Action: reassignAircraft

Parameters:
  flight: Flight
  newAircraft: Aircraft
  reason: string

Rules:
  update Flight.assignedAircraft
  create audit note
  notify maintenance coordinator
  enforce permission checks
```
Apps then call the action:
```ts
await client(reassignAircraft).applyAction({
  flightId: "UA-2401",
  newAircraftId: "N12345",
  reason: "Original aircraft failed maintenance check",
});
```
Palantir's docs describe actions as transactions that change objects, properties, and links, with validations and side effects.

So:
```
Instead of:
  SELECT ...
  JOIN ...
  PATCH /internal-service-x/...
  manually enforce permissions
  manually keep app models aligned with data pipelines

You get:
  client(Flight).where(...)
  flight.assignedAircraft
  client(reassignAircraft).applyAction(...)
  generated types
  central permissions
  shared business actions
```
The cost is that you are now developing inside Palantir’s modeling and runtime world. You get a powerful domain API, but you also accept its abstractions: object types, link types, actions, generated SDKs, Foundry repositories, function publishing, permissions, object storage behavior, and Palantir-specific deployment/versioning workflows.


Q: So it's just a fucking [[Object-Relational Mapping|ORM]]?
A: From a developer's seat, that's not a bad first approximation. 
```
# ORM
database tables <-> application classes

# Palantir's Ontology
enterprise data/models/actions/permissions <-> shared operational domain objects
```
That sounds grandiose, but mechanically it means the model is shared by dashboards, apps, AI agents, actions, permissions, audit logs, and writeback flows.