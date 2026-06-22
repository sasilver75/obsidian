A structured represented of knowledge where things are represented as nodes, and relationships between those things are represented as labeled edges. See [[Graph Database]]s, which operationalize this.
- Node: A thing (book/company/event/location/place/product/idea)
- Edge: A relationship (wrote, founded, causes ,depends on, part of, contradicts)
- Label/Type: Explains what kind of node/relationship something is
- A schema or [[Ontology]] may define allowed types and relationships

A knowledge graph helps answer questions where the useful information is spread across many connected facts.
> Ada Lovelace wrote notes about Babbage’s Analytical Engine and is often considered an early computer programmer.

Can be represented as:
```
Ada Lovelace -> wrote -> Notes on the Analytical Engine
Notes on the Analytical Engine -> about -> Analytical Engine
Analytical Engine -> designed by -> Charles Babbage
Ada Lovelace -> occupation/context -> Mathematician
Ada Lovelace -> associated with -> Early computing
```












