---
aliases:
  - JDL Data Fusion Model
---


References:
- [Article: International Society of Information Fusion: Evolution of the JDL Model](https://isif.org/files/isif/2024-03/ipif-06-01-36.pdf)

The model was conceived in the late 1980’s by the JDL Data Fusion Subgroup, consisting of prominent fusion experts and representatives from various US Government agencies. The model gained considerable influence by its inclusion in Waltz and Llinas's landmark book *Multisensor Data Fusion.*

[[Data Fusion]]: The process of combining data to estimate entity states.
- The specific data fusion problem is that of determining *what data* is relevant to a state estimation problem, and accounting for uncertainty in data relevance, data accuracy, and in the performance of the inference method.

The initial JDL model introduced the notion of fusion "levels", distinguishing classes of fusion processing methods as applicable to major distinguishable classes of problems: Processes that relate to the refinement of estimates or understanding of:
- Objects (level 1)
- Situations (level 2)
- Threats (level 3)
- Processes (level 4)

The original JDL model depicts levels as interacting via  bus architecture, such that processing sequences and access to data are free design variables.

![[Pasted image 20260803093717.png]]



A data fusion model will need an [[Ontology]] and [[Taxonomy]] to clarify such terms as:
- attribute/property/feature/signal/observable
- entity/object/individual/target
- relationship/relation
- structure/complex/situation/scenario
- detection/contact/perceived entity/track

In the late 90s, the model was re-examined to clarify terms, broaden it from its initial focus on tactical military applications, and refine the partitioning scheme:
- L0: Feature/Signal Assessment: Estimation of patterns: paradigmatically signal or feature modulations in 1, 2, or more dimensions, but can extend to most any abstract pattern: numeric or geometric patterns, musical or literary themes, rhyme schemes, etc. 
- L1: Individual Entity Assessment: Estimation of states of entities considered as individuals.
- L2: Situation Assessment: Estimation of relational states and of complexes of relationships.
- L3: Scenario/Impact Assessment: Predictive or forensic estimation of courses of action, scenarios, and outcomes
- L4: System Assessment: Estimating states of the system itself, e.g. sensor and data alignment, estimation or control performance, fidelity of predictive models.

Later a proposal of a DF Level 5, "User Refinement"

These ideas were incorporated in 2004-05 in a isgnificant variant developed by the ISIF Data nd Information Fusion Group (DFIG), as shown in Figure 2.

![[Pasted image 20260803093724.png]]

This DFIG model distinguishes fusion levels as transforming information between entities of various types, effectively partitioning fusion processes on the basis of agency. 

How then do we select among the multitude of JDL model variants and alternatives, which vary in scope, partitioning scheme, and purpose?

We proposed refinements in the early definition of levels (e.g. Figure 3)

![[Pasted image 20260803093732.png]]

The goals were to clarify partitioning and to broaden applicability beyond the tactical military domain.



Figure 4 presents the original, non-hierarchical JDL model, refined and extended to improve clarity and breadth in modeling fusion problems, solutions, and problem domains:
![[Pasted image 20260803093746.png]]









