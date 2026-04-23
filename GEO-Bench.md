June 6, 2023, ServiceNow, TU Munich, Stanford, MIT, ETH Zurich, ASU
[Arxiv Paper](https://arxiv.org/abs/2306.03831)

Standardized benchmark for evaluating geospatial foundation models specifically, filling a gap where existing CV benchmarks (ImageNet, COCO) don't reflect real EO challenges.
- 6 classification tasks
- 6 segmentation tasks

Still (2026) a widely-used standard for GFM comparison; nearly every new [[Remote Sensing|EO]] [[Geospatial Embedding|GFM]] reports Geo-Bench results.

Evaluation protocol:
![[Pasted image 20260419162712.png|500]]

Limitations:
- Image-level labels on some tasks, not always pixel-precise
- No temporal modeling tasks, all datasets are single-timestamp, so models with strong temporal pretraining ([[Prithvi v2]], [[AlphaEarth Foundations|AEF]]) can't demonstrate that advantage.
- No multi-modal tasks: Most tasks are single-sensor, so multi-source models like [[AlphaEarth Foundations|AEF]] don't get credit for radar/LiDAR fusion.

Abstract
> Recent progress in self-supervision has shown that pre-training large neural networks on vast amounts of unsupervised data can lead to substantial increases in generalization to downstream tasks. Such models, recently coined foundation models, have been transformational to the field of natural language processing. Variants have also been proposed for image data, but their applicability to remote sensing tasks is limited. ==To stimulate the development of foundation models for Earth monitoring, we propose a benchmark comprised of six classification and six segmentation tasks, which were carefully curated and adapted to be both relevant to the field and well-suited for model evaluation==. We accompany this benchmark with a robust methodology for evaluating models and reporting aggregated results to enable a reliable assessment of progress. Finally, we report results for 20 baselines to gain information about the performance of existing models. We believe that this benchmark will be a driver of progress across a variety of Earth monitoring tasks.