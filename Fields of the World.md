---
aliases:
  - FTW
---
September 24, 2024, from folks at ASU, Microsoft AI for Good, Taylor Geospatial, WashU
[Arxiv Link](https://arxiv.org/abs/2409.16252)


Abstract
> ==Crop field boundaries== are foundational datasets for agricultural monitoring and assessments but ==are expensive to collect manually==. Machine learning (ML) methods for automatically extracting field boundaries from remotely sensed images could help realize the demand for these datasets at a global scale. However, ==current ML methods for field instance segmentation lack sufficient geographic coverage, accuracy, and generalization capabilities==. Further, research on improving ML methods is restricted by the lack of labeled datasets representing the diversity of global agricultural fields. We present ==Fields of The World (FTW)== -- a ==novel ML benchmark dataset for agricultural field instance segmentation spanning 24 countries on four continents== (Europe, Africa, Asia, and South America). FTW is ==an order of magnitude larger than previous datasets== with ==70,462 samples==, each containing ==instance and semantic segmentation masks paired with multi-date, multi-spectral Sentinel-2 satellite images==. We provide results from baseline models for the new FTW benchmark, show that models trained on FTW have better zero-shot and fine-tuning performance in held-out countries than models that aren't pre-trained with diverse datasets, and show positive qualitative zero-shot results of FTW models in a real-world scenario -- running on Sentinel-2 scenes over Ethiopia.

Specs:
- 24 Countries across Europe, Africa, Asia, South America
	- Diversity matters because average field size varies enormously; in Sub-Saharan Africa, it's ~1.6ha, while in North America, it's ~121ha.
- Total samples: 70,462 chips
- Total area: 166,293 km^2
- Field polygons: 1.63M
- Imagery: [[Sentinel|Sentinel-2]] (RGB + NIR, 10m), two contrasting dates per sample
	- Two dates, chosen to contrast growing vs. off-season periods, helping models distinguish active crop fields from fallow land, forest, etc. Ablations confirmed that adding this second time window improves performance more than adding spectral bands.
- Chip size: 256x256 pixels (1536m x 1536m)
- Label types: Instance mask + 3-class semantic masks (interior, boundary, background)
	- Boundaries as an explicit class (rather than just field/no field) helps models delineate adjacent touching fields, which would otherwise merge. This outperformed 2-class masks in experiments.
- License: CC BY-SA 4.0
- Data hosted on Source Cooperative in [[GeoParquet]] format


![[Pasted image 20260420013456.png]]

