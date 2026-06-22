
A database whose primary model represents facts as a graph: `nodes` represent things, `edges` represent relationships between things, and queries often work by *traversing* those relationships.
Good when:
- Relationships are central to the domain
- Queries involve paths of variable length
- Many-to-many relationships are common
Not ideal when:
- The workload is mostly simple CRUD over records
- Queries are mostly aggregations over huge numbers of rows
- The data is naturally tabular and relationship depth is shallow

While in a [[Relational Database]], relationships are represented indirectly with foreign keys and joins, in a graph database, relationships are first-class records that can have names, direction, types, and properties.
- The main difference is the data model, query ergonomics, and traversal optimization.

```
(Alice)-[:MEMBER_OF]->(Security Team)
(Security Team)-[:OWNS]->(Secrets Service)
(Secrets Service)-[:DEPENDS_ON]->(PostgreSQL)
```

A graph database is useful when the questions you care about are not merely: "What rows match this filter?"
But instead: "How is this thing connected to other things?"

Common graph-shaped questions:
- Who can access this resource, directly or indirectly?
- What services depend on this database?
- Are these two accounts connected through shared devices, addresses, or payments.?
- What's the shortest path between two entities?
- Which customers are similar because of shared behavior?
- What recommendations follow from nearby relationships?

Terms:
- Node/Vertex: An entity, object, or concept (user, company, file, account, server)
- Edge/Relationship: A connection between nodes (`FRIENDS_WITH`, `OWNS`, `DEPENDS_ON`, etc.)
- Property: Data attached to a node or edge (`name`, `created_at`, `risk_score`, `amount`)
- Label/Type: A category for nodes or edges (`User`, `Device`, `Transaction`) 
- Traversal: Following edges from node to node.


A typical graph query starts with an indexed lookup, then expands through edges.
In a property graph query language like Cypher:
```
MATCH (u:User {id: "alice"})-[:MEMBER_OF]->(team)-[:OWNS]->(service)
RETURN service
```
This means:
- Find the `User` node whose `id` is `"alice`"
- Traverse outgoing `MEMBER_OF` relationships to teams
- Traverse outgoing `OWNS` relationships to services
- Return the services
The database is optimized for "walk from here to nearby connected things" queries.


# Main Variants
- [[Property Graph]] Database: Nodes and edges can both have properties. Relationships usually have names and direction. Things like [[Neo4j]], JanusGraph, Amazon Neptune property graph mode
	- `(Alice)-[:WORKS_AT {since: 2021}]->(Acme)`
- [[Resource Description Framework]] (RDF) triple store: GraphDB, Blazegraph, Apache Jena, Amazon Neptune RDF mode
	- `Alice worksAt Acme`

See also: [[Web Ontology Language]] (OWL)
















