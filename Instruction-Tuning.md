---
aliases:
  - Instruction-Tune
  - Instruction-Tuned
  - Instruction Fine-Tuning
  - IFT
---
# Considerations of Creating/Evaluating Instruction Data
- ==Mixing few-shot settings==: Training with mixed zero-shot and few-shot prompts significantly improve performance in both settings.
- ==Task diversity==: Large models benefit from continuously increasing the number of tasks.
- ==Data augmentation==: Augmenting the data such as by inverting inputs/outputs (eg turning a question answering task into a question generation task) is beneficial.
- ==Mixing weights==: When using a combination of instruction-tuning dataset, appropriately tuning the mixing weights is important.
# Questions you should ask your IFT dataset
1. ==Data source==: How was this data obtained? Many have been generated using ChatGPT, meaning they inherit biases of the source model, or might noisy. Human-written examples are more expensive to obtain but are more high-quality.
2. ==Data quality==: Was any filtering done to improve the quality of the generated data? In most cases, filtering is based on simple heuristics or a pre-trained model, which can result in noisy data.
3. ==Domain and language coverage==: Most datasets cover general QA-style use cases and are in English, but similar methods can be used to obtain data in other domains or languages.
4. ==Number of dialog turns==: A "dialog turn" is an utterance by one speaker. Most datasets are single-turn; they consist of a prompt and a single response. Multi-turn may be necessary to train a more conversational model.
5. ==License terms==: Data generated using OpenAI models is subject to OpenAI's terms of use, which prohibit using the data to develop competing models. So look for data with a more permissive license to avoid legal complications!
# More Tips
- ==Quality > Quantity==: As [[LIMA]] showed, training on sets of (smaller, even) high-quality data outperforms instruction-tuning on larger, noisier data. Using more diverse prompts and quality filtering both improve performance too.
- ==Imitation != Mastery==: Models that are instruction-tuned on ChatGPT-generated data mimic ChatGPT's style (and thus might fool human raters), but *not* its factuality (The False Promise of Imitating Proprietary LLMs, 2023), performing worse on standard benchmarks. Using stronger base models is the best way to address this.
- ==The stronger the base, the better==: More powerful base models also produce stronger instruction-tuned models ([[Tulu]] using a 65B model)
- ==The combination wins!==: Combining multiple instruction-tuning datasets results in the best average performance across tasks ([[Tulu]]). Dataset mixing and developing modular instruction-tuned models are thus important research directions.