---
aliases:
  - Cypher
---


A [[Graph Database]] system designed to store and query data as nodes, relationships, and properties, instead of primarily as rows and tables. For data where the connections are as important as the individual records.
- Neo4j is not “better than SQL databases” in general. It is better for certain graph-shaped workloads. A relational database is often still the better default for tabular business records, aggregations, reporting, strict normalized schemas, and workloads where relationships are simple or shallow.

```cypher
(:Person {name: "Maya"})-[:WORKS_AT]->(:Company {name: "Acme"})
(:Person {name: "Maya"})-[:USES]->(:Device {id: "phone-17"})
(:Device {id: "phone-17"})-[:USED_FOR]->(:Login {ip: "203.0.113.9"})
```

In Neo4j, the relationship `WORKS_AT` isn't just a foreign key value, it's a first-class stored object that can have its own type, direction, and properties.

A Neo4j database that uses the ***labeled property graph*** model:
- ==Node==: An entity, such as a person, account, product, city, server, or document
- ==Label==: A category attached to a node, such as `Person`, `Company`, or `Transaction`
- ==Relationship==: A directed connection between two nodes, such as `FRIENDS_WITH`, `BOUGHT`, etc.
- ==Relationship type==: The kind of relationship
- ==Property==: A key-value attribute on a node or relationship

Components:
- Neo4j: The core graph database engine.
- Cypher: The graph query language.
- Neo4j Browser: Interactive query UI, often used by developers.
- Neo4j Bloom: Visual graph exploration tool, often used by analysts.
- APOC: A library of extended procedures and utilities.
- Graph Data Science: Neo4j tooling for graph algorithms and machine learning workflows.
- Aura: Neo4j's managed cloud offering


A Neo4j is usually queried with ==Cypher==, a graph query language whose syntax visually resembles graph patterns.
```cypher
MATCH (p:Person {name: "Maya"})-[:WORKS_AT]->(c:Company)
RETURN c.name
```
- The query means: find a `Person` node named Maya, follow the outgoing `WORKS_AT` relationship, and return the connected `Company`.
- The problem Neo4j solves is that many real systems are relationship-heavy. In a relational database, asking "which users are connected to this suspicious account within three hops" might require several recursive joins or complex common table expressions. In Neo4j, that kind of traversal is the natural shape of the database.
- Mechanically, Neo4j often works by first using an index to find a starting node, then traversing stored relationships from that node.


A node can have multiple labels:
```cypher
(:Equipment:GroundVehicle {name: "M1A2 Abrams"})
```
A relationship can have properties too:
```cypher
(:Platform)-[:HAS_COMPONENT {
  source: "maintenance_manual_2024",
  confidence: 0.92,
  validFrom: date("2021-01-01")
}]->(:Component)
```
This matters a lot in knowledge graph, because many relationships are not timeless, universally-certain facts; they may be sourced, inferred, time-bounded, or contested.


Cypher is built around graph pattern matching:
```cypher
(:Platform)-[:HAS_COMPONENT {
  source: "maintenance_manual_2024",
  confidence: 0.92,
  validFrom: date("2021-01-01")
}]->(:Component)
```
"Find a `Platform` node with that name, follow outgoing `HAS_COMPONENT` relationships, and return the connected `Components`"
![[Pasted image 20260622141528.png]]

A multi-hop query might look like:
```cypher
MATCH path = (p:Platform {name: "M1A2 Abrams"})-[*1..3]-(related)
RETURN path
```
That asks for things within one to three relationship hops of the platform.
It's a powerful thing that he query syntax actually resembles the shape of the question. The dangerous part heres is that "connected within three hops" doesn't automatically mean "meaningfully related."


```cypher
MATCH (supplier:Organization {name: "Example Supplier"})
MATCH (supplier)<-[:SUPPLIED_BY]-(component:Component)<-[:HAS_COMPONENT]-(platform:Platform)
RETURN platform.name, component.name
```
This finds platforms that are effected by a supplier.

```Cypher
// We can add comments 
CREATE (:User { anem: "@jeff" }) -[:FOLLOWS]->(:User { name: "@neo" })

// we can add constraints
CREATE CONSTRAINT ON (user:User) ASSERT user.name IS UNIQUE

// Can also define local variables in the query and then return them from the statement
CREATE (j:User { anem: "@jeff" })-[r:FOLLOWS]->(n:User { name: "@neo" })
RETURN j, r, n

// We can connect multiple tweet nodes to a user
CREATE (:User { name: "2alice" })-[:SAYS]->(:Tweet {
	text: "hi mom",
	created: date("2023-011-02")
})

// We can read all tweets from users I follow.
// Note that the -- means "match one relationship of any type, in either direction". Frankly this is probably overly-broad for what the query is trying to do. since it might match on relationships like `DOWNVOTED`
MATCH (u:User {name: "@jeff"})-[:FOLLOWS]->(:User)--(t:Tweet)
WHERE t.created > date("2023-01-01")
// WHERE t.text =~ "(?i)Hi.*"   // Or where the tweet text matches a condition
// WHERE NOT (u)-[:MUTED]->(:User)  // Or from users that our u User hasn't muted
RETURN t.text


// Finding any tweet reachable in 1-3 relationship hops, using any relationship type, in either direction
MATCH path = (u:User {name: "@jeff"})-[*1..3]-(t:Tweet)
RETURN
  t.id AS tweetId,
  length(path) AS hops,
  [rel IN relationships(path) | type(rel)] AS relationshipTypes


// Find tweets connected to jeff by exactly oen relationship, in either direction, and bind that relationship to the varaible r.
MATCH (u:User {name: "@jeff"})-[r]-(t:Tweet)
RETURN
  t.id AS tweetId,
  type(r) AS relationshipType,
  properties(r) AS relationshipProperties
```

Relationship patterns:
- `--`: Any single relationship, any type, either direction
- `-[r]-`: Any relationship, either direction, bound to variable `r`
- `-[:POSTED]->`: Outgoing `POSTED` relationship
- `<-[:POSTED]-`: Incoming `POSTED` relationship
- `-[:POSTED]-`: `POSTED` relationship in either direction


### Common Use Cases

| Use case                        | Why graph storage helps                                                        |
| ------------------------------- | ------------------------------------------------------------------------------ |
| Fraud detection                 | Fraud rings are often visible as suspicious connection patterns                |
| Recommendation systems          | “People who bought X also bought Y” is a graph traversal problem               |
| Identity and access management  | Users, roles, groups, permissions, and resources form a graph                  |
| Knowledge graphs                | Concepts and facts are connected explicitly                                    |
| Network and dependency analysis | Services, machines, packages, or infrastructure dependencies are relationships |
| Social graphs                   | Friends, followers, memberships, and interactions are naturally graph-shaped   |


A few deeper points matter a lot:
- A knowledge graph can be an RDF/OWL semantic web graph with formal ontologies, triplets, inference rules, and standardized vocabularies. Sometimes it means a property graph in Neo4j, where nodes/relationships have labels and properties, but semantics are mostly enforced by application code, conventions, constraints, and queries.
- Neo4j is "schema-flexible," not meaningfully "schema-free." You can add different labels and relationship without predefining every table shape, but a serious knowledge graph still needs a domain model. Otherwise the graph becomes a pile of loosely connected facts with inconsistent names.
- The thing to watch for is that graph modeling shifts difficulty around. Neo4j makes relationship traversal natural, but the hard part becomes deciding what the relationships mean.

Paying attention to concepts like:
- ==Ontology/domain model==: Declines what entity types/relationships are allowed to mean.
- ==Provenance==: Records where each fact came from, such as document/database/analyst/sensor feed.
- ==Confidence==: Distinguishes known facts from inferred, uncertain, stale, or contradictory facts.
- ==Temporality==: Handles facts that are true only during certain time ranges.
- ==Identity resolution==: Decides whether two names or records refer to the same real-world thing.
- ==Classification/access control==: Controls who can see which nodes, relationships, and properties.
- ==Graph traversal==: Finds nearby related facts through paths.
- ==Path semantics==: Determines whether a multi-hop path actually means something valid.
- ==Constraints and indexes==: Keep the graph queryable and prevent some bad data.
- ==Graph algorithms==: Finds central entities, clusters, shortest paths, dependencies, and influence patterns.
















___________

In context of [[Smack Technologies]], a knowledge graph might represent things like:
```
(:Platform {name: "M1A2 Abrams"})
  -[:HAS_COMPONENT]->
(:Component {name: "AGT1500 turbine engine"})

(:Platform {name: "M1A2 Abrams"})
  -[:HAS_CAPABILITY]->
(:Capability {name: "armored maneuver"})

(:Component {name: "AGT1500 turbine engine"})
  -[:SUPPLIED_BY]->
(:Organization {name: "Honeywell"})

(:Platform {name: "M1A2 Abrams"})
  -[:OPERATED_BY]->
(:MilitaryUnit {name: "1st Armored Brigade"})
```

Maybe useful for questions like:
```
Which platforms depend on a component supplied by this vendor?

Which equipment supports this mission profile and is compatible with this communications system?

If this part becomes unavailable, which systems, units, or missions are affected?

Which documents provide evidence for this claimed capability?

What equipment is within two relationship hops of this threat system?

Which platforms share enough components or capabilities that maintenance knowledge transfers between them?
```











