---
aliases:
  - OWL
---
A formal [[Ontology]] language built for machine reasoning over [[Resource Description Framework|RDF]]-style data. Used for defining precise logical meanings, classes, constraints, and inferable facts.
- The odd acronym order is historical.

> OWL is a standardized logic language for turning a knowledge graph from a collection of facts into a system that can infer, classify, and check meanings.

A formal language for defining the meaning of classes, relationships, and logical constraints so that a reasoner can infer additional facts. It's more of an ontology/logic language than it is a data model like something like [[Labeled Property Graph]] or [[Resource Description Framework|RDF]].

```
Every Tank is a GroundVehicle.
Every GroundVehicle is a Platform.
If something has an Engine as a component, then it is a PoweredSystem.
No Organization can also be a PhysicalComponent.
```
So a reasoner can infer things like
```
AGT1500_Engine is a Component.
M1A2_Abrams is a PoweredPlatform.
```
Think of OWL as a way to build a formal domain theory.
Not just "Here are the entities," but:
- Here are the kinds of entities that can exist.
- Here are the relationships that can connect them.
- Here are the logical rules that those relationships obey.
- Here are the categories that overlap.
- Here are the categories that cannot overlap.
- Here are the facts that can be inferred from other facts.

OWL says things like:
```
If something is in this category, then it is also in that category.
If two relationships exist, another relationship follows.
If an entity has this kind of relationship, it belongs to this class.
These two categories cannot overlap.
This relationship has an inverse relationship.
```
OWL exists because RDF triples alone do not say enough about meaning.
- RDF might say `A hasComponent B`
- OWL can say:
	- What hasComponent means.
	- What kinds of things can be components.
	- What follows if something has a component.
	- Which categories imply other categories.
	- Which combinations are impossible.

An [[Ontology]] in this context is the formal model: The classes, properties, and axioms that define the domain.

| OWL concept     | Meaning                                                                     |
| --------------- | --------------------------------------------------------------------------- |
| Class           | A category of things, such as `Tank`, `Platform`, or `Component`            |
| Individual      | A specific entity, such as `M1A2_Abrams`                                    |
| Object property | A relationship between two individuals, such as `hasComponent`              |
| Data property   | A relationship from an individual to a literal value, such as `hasWeightKg` |
| Subclass        | A class hierarchy relationship, such as `Tank` subclass of `GroundVehicle`  |
| Restriction     | A logical rule about class membership or relationships                      |
| Reasoner        | Software that derives implied facts or detects logical inconsistency        |









