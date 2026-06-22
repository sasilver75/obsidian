---
aliases:
  - RDF
---
A [[World Wide Web Consortium|W3C]] standard data model for representing facts as triples. Interoperable semantic data exchange across systems.

Typically:
```
subject predicate object
```

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


