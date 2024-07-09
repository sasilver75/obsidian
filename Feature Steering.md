


Related concepts:
- ==[[Feature Steering]]==: A broader concept that includes steering vectors. The aim being to guide model outputs by manipulating internal features or representations.
- ==[[Feature Clamping]]==: A specific technique within Feature Steering, involving fixing or "clamping" certain features to specific values during inference. While Steering Vectors allow for more gradual, continuous influence, Feature Clamping is more binary; either fixed or not.
- ==[[Steering Vector|Concept Vector]]==: Representations of specific concepts, attributes, or semantic features in the embedding space of a neural network.
- [[Steering Vector|Control Vector]]/[[Steering Vector]]: The use of some (eg scaled) concept vector(s) to influence the internal representations of a model.

((It seems to me that a steering vector is just a control vector in the context of language modeling, and that we use a concept vector as a steering vector. This is a different technique from Feature Clamping, but both are within the toolbox of Feature Steering, which is a problem/cause area within the subfield of Representation Engineering))