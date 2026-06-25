
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
- [[Labeled Property Graph]] Database: Nodes and edges can both have properties. Relationships usually have names and direction. Things like [[Neo4j]], JanusGraph, Amazon Neptune property graph mode
	- `(Alice)-[:WORKS_AT {since: 2021}]->(Acme)`
- [[Resource Description Framework]] (RDF) triple store: GraphDB, Blazegraph, Apache Jena, Amazon Neptune RDF mode
	- `Alice worksAt Acme`

See also: [[Web Ontology Language]] (OWL), [[Ontology]]


![[Pasted image 20260624120103.png]]



# Query Languages

> One of the things that I attribute the immaturity of the space is to the fact that there isn't just a single graph query language. There's an effort going on now to standardize query languages, especially across labeled property graphs, but that's just starting now, so it will be several years.
> - David Bechberger (2020), talking about (I think?) [[Graph Query Language]]




# Comments

> A mistake I've seen customers make all the time... really understand what you're looking for in your data. People will put all their relationships in a graph... and somehow think that magic is going to happen and you'll get insight out. You have to know what questions you're going to ask. Is a question going to require you to look at all the data in your graph? If it does, then you're probably not going to get a 20ms response back. Is it instead going to look at a subgraph of your data? Understand what it is you're looking for and what your question takes to answer...

> Understand what your algorithm is doing. If you want to do a vertex/edge lookup, that's a quick thing. If you want to understand the degree of a vertex, that's also quick. But you typically want to do other things with a graph, things like Shortest Path, PageRank, Chromatic Number, All Paths. These might take a lot of time.

> Understand your data. Customers often don't take the time to understand the complexity of the data that they work on. The branching factor of a graph (the number of successors for any node) is something that often bites. If you have a branching fact or of 2, then as you traverse it's 1->2->4->8. This doesn't seem like a big deal, until you have a branching factor of 6... if you want to go out 6 levels deep, you're touching 56,000 vertices to get there.

> Understand what the tails of your distribution are as well. Sometimes you have a small number of nodes that extremely high degree.

> You can't just apply relational thinking to a graph; graphs are all about relationship. The graphs space and tooling are immature, relative to the relational space. Don't expect a seamless transition.











