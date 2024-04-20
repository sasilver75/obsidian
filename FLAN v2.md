Jan 31, 2023 -- [[Google Brain]]
Paper: [The Flan Collection: Designing Data and Methods for Effective Instruction-Tuning](https://arxiv.org/abs/2301.13688)
HuggingFace Dataset card: [FLAN v2](https://huggingface.co/datasets/philschmid/flanv2)

An investigation into ablations around the training dataset used for [[FLAN-T5]], with the final datasets being released.

Interesting: Training with [[Zero-Shot Prompting]], [[Few-Shot Learning]], [[Chain of Thought]] results in strong performance.

Abstract
> We study the design decisions of publicly available instruction tuning methods, and break down the development of Flan 2022 (Chung et al., 2022). Through careful ablation studies on the Flan Collection of tasks and methods, ==we tease apart the effect of design decisions which enable Flan-T5 to outperform prior work== by 3-17%+ across evaluation settings. We find task balancing and enrichment techniques are overlooked but critical to effective instruction tuning, and in particular, ==training with mixed prompt settings (zero-shot, few-shot, and chain-of-thought) actually yields stronger (2%+) performance in all settings==. In further experiments, we show Flan-T5 requires less finetuning to converge higher and faster than T5 on single downstream tasks, motivating ==instruction-tuned models as more computationally-efficient== starting checkpoints for new tasks. Finally, to accelerate research on instruction tuning, we ==make the Flan 2022 collection of datasets, templates, and methods publicly available== at [this https URL](https://github.com/google-research/FLAN/tree/main/flan/v2).




