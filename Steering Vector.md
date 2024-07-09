---
aliases:
  - Control Vector
  - Concept Vector
---


A specific application/technique in the broader field of [[Representation Engineering]], a subfield aiming to understand and manipulate the internal representations of neural networks by directly intervening in the model's hidden states or activations.

A steering vector is one of these high-dimensional vectors in representation that represents a specific direction or attribute of text generation. The idea is to find control vectors that influence the model's output towards certain characteristics or styles and add these to the model's internal representations (either at the input embeddings, hidden states, attention mechanisms, or output logits), nudging model predictions toward desired attributes.
- Formal vs Casual
- Positive, Negative, Neutral
- Creativity vs Factuality
- Focus on specific topics or domains
Multiple steering vectors can even be combined to control different aspects simultaneously!
Can be seen as a form of fine-tuning, and is often used in [[Parameter-Efficient Fine-Tuning|PEFT]] techniques.

Often created by taking the difference between embeddings of text *with* and *without* some desired attribute.

Related concepts:
- ==[[Feature Steering]]==: A broader concept that includes steering vectors. The aim being to guide model outputs by manipulating internal features or representations.
- ==[[Feature Clamping]]==: A specific technique within Feature Steering, involving fixing or "clamping" certain features to specific values during inference. While Steering Vectors allow for more gradual, continuous influence, Feature Clamping is more binary; either fixed or not.
- ==Concept Vectors==: Representations of specific concepts, attributes, or semantic features in the embedding space of a neural network.

((It seems to me that a steering vector is just a control vector in the context of language modeling, and that we use a concept vector as a steering vector. This is a different technique from Feature Clamping, but both are within the toolbox of Feature Steering, which is a problem/cause area within the subfield of Representation Engineering))