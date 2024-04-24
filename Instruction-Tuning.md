---
aliases:
  - Instruction-Tune
  - Instruction-Tuned
  - Instruction Fine-Tuning
---

# Important Aspects of Instruction Data
- ==Mixing few-shot settings==: Training with mixed zero-shot and few-shot prompts significantly improve performance in both settings.
- ==Task diversity==: Large models benefit from continuously increasing the number of tasks.
- ==Data augmentation==: Augmenting the data such as by inverting inputs/outputs (eg turning a question answering task into a question generation task) is beneficial.
- ==Mixing weights==: When using a combination of instruction-tuning dataset, appropriately tuning the mixing weights is important.


(See InstructGPT Paper; possibly pre-RLHF)