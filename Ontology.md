---
aliases:
  - Ontologies
  - Ontological
---

References:
- Video: [What is Ontology](https://youtu.be/UW57RW-4kWs?si=zp5pZO72ughMjZ1s)?

A *structured model* of the kinds of things that exist in a domain and the relationships that can hold between them. A formal or semi-formal model of a domain’s concepts, relationships, and rules, used to make knowledge consistent, explicit, reusable, and sometimes machine-reasonable.
- In Philosophy: The study of what exists: entities, categories, being existence.
- In Computer Science: A formal or semi-formal model of the entities, categories, relationships, and rules in a domain.

> "Formally well-defined machine-interpretable controlled vocabularies designed to represent entities and logical relationships among them. They make explicit the implicit meanings buried in datasets, by using basic principles of formal logic."

> "An Ontology is what our business/technology needs to know about, and how those things relate to one another. Not just what exists, but how the things that exist interrelate. May be domain-specific. The format isn't essential; it could be embodied in a [[Knowledge Graph]], or it could just be a PDF. A good ontology is all of a Language, a Graph (mathematical object), and a Model of Reality."


Answers things like:
- What kinds of entities exist?
- What categories do those entities belong to?
- What relationships are meaningful or even allowed?
- What constraints or rules apply?
- When are two terms the same, different, broader, narrower, or incompatible?

A [[Knowledge Graph]] defines Facts, while the Ontology defines the conceptual structure that those facts are supposed to follow. An ontology is the model, while the knowledge graph is populated data.
```
Ontology:
Person is a type of Entity.
Company is a type of Organization.
founded is a relationship from Person to Organization.
headquartered in is a relationship from Organization to Place.

Knowledge graph:
Steve Jobs -> founded -> Apple Inc.
Apple Inc. -> headquartered in -> Cupertino.
Cupertino -> located in -> California.
```
Without an ontology, different people or systems might describe the same domain inconsistently.
One system might say `Employee -> works for -> Company` while another says `Person -> employed by -> Organization`. These may mean the same thing, slightly different things, or completely different things. The ontology makes the intended meaning explicit.

==Think of an Ontology as a domain blueprint.==

### Main parts of Ontologies

|Part|Meaning|Example|
|---|---|---|
|Class / Type|A category of thing|`Person`, `Disease`, `Medication`|
|Instance / Individual|A concrete thing|`Ada Lovelace`, `Aspirin`, `Diabetes Mellitus`|
|Relationship / Property|A meaningful connection|`treats`, `causes`, `located in`|
|Attribute / Data property|A value attached to an entity|`birth date`, `dosage`, `temperature`|
|Hierarchy|Broader/narrower category structure|`Cardiologist` is a subtype of `Physician`|
|Constraint|A rule limiting valid data|A `birth date` must be a date|
|Axiom|A formal statement that enables reasoning|Every `Cardiologist` is a `Physician`|
|Equivalence|Two terms mean the same thing|`Heart Attack` same as `Myocardial Infarction`|
|Disjointness|Two categories cannot overlap|`Living Person` and `Deceased Person`|

### Formal vs Informal Ontologies
|Kind|Description|Example|
|---|---|---|
|Informal ontology|Human-readable categories and relationships|A glossary with relationship notes|
|Semi-formal ontology|Structured but not fully logical|A JSON schema plus domain definitions|
|Formal ontology|Machine-interpretable logic|RDF Schema, OWL, description logic|


Ontologies are powerful because they make meaning explicit; they're especially useful when multiple systems, teams, or datasets need to agree on shared concepts.

But Ontologies have costs:
- They require careful modeling.
- They can become too abstract.
- They can become outdated as the domain changes.
- Different stakeholders may disagree about categories.
- Overly rigid ontologies can make real-world exceptions hard to represent.
- Under-specified ontologies can become vague diagrams with little practical value.


________

The [[Palantir Ontology]] system is "an operational digital twin layer that maps an organization's raw data from ERPs, CRMs, and IoT sensors into real-world business concepts (objects, properties, and links). 

