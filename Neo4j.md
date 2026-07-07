---
aliases:
  - Cypher
---
Todo:
- Look at PRotege Plugin for ontologies
- https://www.reddit.com/r/semanticweb/comments/1hhgnpw/a_good_resource_to_learn_about_ontologies_from/ 
- https://www.reddit.com/r/OntologyEngineering/
- https://www.reddit.com/r/semanticweb/comments/1s011z4/is_learning_ontology_development_still_worth_it/
- 


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


___________________

[Introduction to Neo4j and Graph Databases - M David Allen (Partner Solution Architect @ Neo4J), 2019](https://youtu.be/oRtVdXvtD3o?si=Nhx95BJ1UcNwfaS0)




__________

[Neo4J Crash Course](https://youtu.be/8jNPelugC2s?si=KNecvVsKvHuuOcJ3) (2022)




_______________

Conversation with Codex about data modeling in Neo4j and [[Labeled Property Graph]] databases... about the best practices as well as pitfalls.

The main skill is deciding what deserves identity as a node, what should be a relationship, and what should stay as a property.

Neo4J's guidance is strongly query-first:
- Identify the questions that the application must answer
- Build an initial graph model
- Test the queries and performance
- Refactor the model as the use cases change

See this structure:
```cypher
(:Customer {customerId: "C123"})
  -[:PLACED {at: datetime("2026-06-24T10:30:00Z")}]->
(:Order {orderId: "O456"})
  -[:CONTAINS {quantity: 2, unitPrice: 19.99}]->
(:Product {sku: "SKU-9"})
```
This graph is good when the application is asking relationship-shaped questions:
```cypher
MATCH (c:Customer {customerId: $customerId})-[:PLACED]->(o:Order)-[line:CONTAINS]->(p:Product)
RETURN o.orderId, p.sku, line.quantity, line.unitPrice
```
This graph would be less useful if most queries were simple table scans, aggregate reports, or isolated key-value lookups.

NOTE:
- Relationships always have a singular direction (i.e. unidirectional, not bidirectional); each relationships has a start node, an end node, a type, and properties.
	- Still, some domain facts are going to be bidirectional, e.g. `Sam --Friend_Of--> Roz`. In this case, the domain fact is bidirectional, but in the database it's still stored in a single unidirectional edge. You typically want to have a deterministic rule/canonicalization choice of how you make this edge, such as storing the relationship from the lower `PersonId` to the higher `PersonId`, and avoid doubling-up on edges, in the case that both relationships mean the same domain fact.
		- A common pitfall is choosing direction based on who initiated the friendship request., which usually mixes two facts; instead, you'd want to have a `(:Person)-[:SENT]->(:FriendRequest)-[:TO]->(:Person) and (:Person)-[:FRIEND_OF]->(:Person)`


Start with questions, not tables. A graph model should be shaped around queries like "which customers share payment instruments?" "What services are affected by this outage?" or "Which products are frequently bought together?" This differs from relational normalization, where the model is often shaped round eliminating redundancy first.


Use meaningful relationship types:
- Prefer `(:Person)-[:WORKED_AT]->(:Company)` over the more vague (:Person)-[:RELATED_TO]->(:Company {kind: "employment"})`. Relationships should carry domain meaning.

Use intermediate nodes for contextual facts.
- Employment is often not merely `(:Person)-[:WORKED_AT]->(:Company)`
- If you need role, start date, end date, manager, compensation band, or overlapping employment analysis, model employment itself as a node:
```cypher
(:Person)-[:HELD]->(:Employment {startDate, endDate})
(:Employment)-[:AT]->(:Company)
(:Employment)-[:AS]->(:Role)
(:Employment)-[:MANAGED_BY]->(:Person)
(:Employment)-[:IN_PAY_BAND]->(:PayBand)
(:Employment)-[:LOCATED_IN]->(:Office)
```
Q: Why would we ant to have this as a separate node, instead of having it as an "employed_at" edge with properties?
A: Neo4j relationships themselves cannot have relationships to other nodes.  The can have properties, but those properties are terminal values. So if `manager`, `role`, or `payBand` is just a  value, put it on the relationship, but if `manager`, `role`, or `payBand` should connect to other nodes, then make the employment itself a node.


Create constraints early. 
- Use uniqueness constraints for business identifiers such as `Customer.customerId`, `Order.orderId`, and `Product.sku`.

Use indexes as entry points.
- Neo4j traversals are fast after the starting node is found, but Neo4j still needs a good way to find the starting node.
- Index common lookup properties and high-selectivity filters.

Profile your real queries, using `EXPLAIN` to see the planned query without running it and `PROFILE` to run the query and see actual database hits.

The biggest pitfall is building a relational model with graph syntax. If nodes contain foreign-key-like ID arrays such as `friendIds: ["u1", "u2"]`, the model is hiding graph structure inside properties. You should use relationships instead.

Another pitfall is making *everything* a node. Something like `birthDate` usually belongs as a property. A `Date` node is useful only when dates are shared traversal anchors, part of calendar hierarchy, or part of a query pattern.

Dense "supernodes" are dangerous. A node like `(:Country {code: "US"}` that's connected to 200 million users can become an expensive fan-out point. Sometimes that's correct, but often you need bucketing, more selective relationships, or a different query entry point.

Unbounded variable length-paths can explode. Queries like this can match enormous numbers of paths:
```cypher
MATCH p = (a)-[:CONNECTED_TO*]->(b)
RETURN p
```
Prefer bounded traversal and inline predicates:
```cypher
MATCH p = (a)-[:CONNECTED_TO*1..4]->(b)
RETURN p
```
Neo4j's Cypher docs warn that broad, quantified path patterns can match very large numbers of paths and recommended precise predicates.
Don't model for the diagram. A pretty graph visualization can still perform badly or answer the wrong questions. The real test is whether the model makes the important Cypher queries simple, selective, and stable.

Practical rule: ==A good Neo4j model makes the most improtant query read almost like a sentence:==
```cypher
MATCH (customer:Customer {customerId: $id})
      -[:PLACED]->(order:Order)
      -[:CONTAINS]->(product:Product)
RETURN order, product
```
The real test is whether the model makes the important Cypher queries simple, selective, and stable.
If the Cypher is awkward, full of property decoding, generic relationship filtering, or huge variable-length traversals, the model is probably telling you where it wants to be refactored.



Q: Is Neo4j a sort of "schemaless" database like Mongo, where you can just stuff bullshit into there and it's on the applications to have to deal with this when they pull it out?
A: A better phrase is schema-optional; nodes and relationships *can be written freely by default,* but you can add schema-on-write enforcement with constraints.

Neo4j supports constraint on both nodes and relationships:
```cypher
// Unique user id
CREATE CONSTRAINT user_id
FOR (u:User) REQUIRE u.id IS UNIQUE;

// Required user name
CREATE CONSTRAINT user_name_required
FOR (u:User) REQUIRE u.name IS NOT NULL;

// User name must be a string (note that this itself does not mean required)
CREATE CONSTRAINT user_name_type
FOR (u:User) REQUIRE u.name IS :: STRING;

// Relationships having a property required
CREATE CONSTRAINT purchase_at_required
FOR ()-[p:PURCHASED]-() REQUIRE p.at IS NOT NULL;

// Relationship property type
CREATE CONSTRAINT purchase_at_type
FOR ()-[p:PURCHASED]-() REQUIRE p.at IS :: ZONED DATETIME;
```
The core constraint categories are:
- Property uniqueness: If the property exists, its value must be unique for that label or relationship type
- Property existence: The property must exist
- Property type: If it exists, it must have the specified Cypher type
- Node key/relationship key: The property must exist and be unique, like a required business key

Note: A type constraint alone does *not* mean "required," it means "if present, must have this type."

Neo4j constraints do not normally define an exact closed set of allowed properties. If `:User` has constraints on `id`, `email`, and `createdAt, then Neo4j will still allow:
```cypher
CREATE (:User {
  id: "u1",
  email: "ada@example.com",
  createdAt: datetime(),
  randomExtraProperty: "allowed unless separately controlled"
})
```
So it doesn't behave like Postgres where all allowed columns are declared, and undeclared columns are impossible.






























