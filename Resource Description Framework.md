---
aliases:
  - RDF
  - RDF TripleStore
---
A [[World Wide Web Consortium|W3C]] standard data model for representing facts as triples. Interoperable semantic data exchange across systems.

Based on a model:
```
subject predicate object
```
- Each property is stored as a specific vertex in your graph.
	- If you had, in the relational world, a row of data with 4 columns, you would end with 5 vertices in your graph with 4 edges connecting them.
- They have an inference engine... you put in these triplets, and then put in rules about your data. This inference engine then runs on top of it and is able to create new relationships based on these roles.


Example:
```
:M1A2_Abrams :hasComponent :AGT1500_Turbine_Engine .
:AGT1500_Turbine_Engine :manufacturedBy :Honeywell .
```

Each RDF triple says one fact:
```
M1A2 Abrams hasComponent AGT1500 trubine engine
AGT1500 turbine engine manufacturedBy Honeywell
```

The mental model is a "global web of statements."

RDF strongly emphasizes global identifiers, usually ==Internationalized Resource Identifiers== (IRIs), An IRI is a globally meaningful identifier, similar in spirit to a URL.
- That matters because RDF is designed for merging data from different sources. Two systems can both say things about the same IRI, and those statements can be combined into one graph.

RDF is usually queried using using [[SPARQL]], rather than [[Neo4j|Cypher]]:
```SPARQL
SELECT ?component
WHERE {
  :M1A2_Abrams :hasComponent ?component .
}
```


