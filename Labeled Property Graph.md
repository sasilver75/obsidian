---
aliases:
  - LPG
  - Property Graph
---
A graph data model with nodes, relationships, labels/types, and key-value properties. Its main job is practical graph storage and querying, especially in systems like [[Neo4j]]. LPGs are good for traversing patterns in your Graph that you already know exist. If you instead want to deal with inference more, perhaps go to the [[Resource Description Framework|RDF]] side.¿
- See also: [[Resource Description Framework]] (RDF) and [[Web Ontology Language]] (OWL)
	- LPGs are probably the more common type of graph database, and perhaps are most similar to relational databases. You have Vertices/Nodes, which represent your entities, you have Edges which represent relationships, and you have Attributes which represent the attributes on your data, which are stored as KV pairs on vertices and edges. A unique property is that the Edges can have attributes on them themselves. 
- 

Represents data as nodes connected by relationships, where both nodes and relationships can have properties.
```cypher
(:Platform {name: "M1A2 Abrams"})
  -[:HAS_COMPONENT {source: "manual-123", confidence: 0.93}]->
(:Component {name: "AGT1500 turbine engine"})
```

In a property graph: 
- Node: Entity, such as a platform, component, organization, document, or capability.
- Label: Category attached to a node, such as `Platform` or `Component`
- Relationship type: Directed connections between two nodes
- Relationship: Meaning of the connection, such as `HAS_COMPONENT`
- Property: Key-value data on a node or relationship

Property graphs are ergonomic for application developers. You can attach data directly to relationships such as `validFrom`, `confidence`, `source`, or `classification`.

Example Cypher query:
```cypher
MATCH (p:Platform)-[r:HAS_COMPONENT]->(c:Component)
WHERE p.name = "M1A2 Abrams"
RETURN c.name, r.confidence
```
- Often the easiest model when your team wants to build software around graph-shaped domain data.






