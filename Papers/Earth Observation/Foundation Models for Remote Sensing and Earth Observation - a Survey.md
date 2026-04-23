October 2024 (last revised v3 June 2025; note that this main paper was written before the recent crop of [[Clay]], [[AlphaEarth Foundations|AlphaEarth]], [[OlmoEarth]])
[Arxiv Link](https://arxiv.org/pdf/2410.16602)

Takeaway: ...
- ==No RSFM has yet achieved the zero-shot generalization that makes general FMs transformative.==
- TAKEAWAY: ==The most pressing bottleneck is data==: More diverse, higher-resolution, multi-modal, temporally rich datasets with automatic annotation pipelines.

_________

Abstract
> [[Remote Sensing]] (RS) is a crucial technology for observing, monitoring, and interpreting our planet, with broad applications across geoscience, economics, humanitarian fields, etc. While artificial intelligence (AI), particularly deep learning, has achieved significant advances in RS, unique challenges persist in developing more intelligent RS systems, including the complexity of Earth's environments, diverse sensor modalities, distinctive feature patterns, varying spatial and spectral resolutions, and temporal dynamics. Meanwhile, recent breakthroughs in large [[Foundation Model]] (FMs) have expanded AI's potential across many domains due to their ==exceptional generalizability and zero-shot transfer capabilities==. However, their ==success has largely been confined to natural data like images and video==, with degraded performance and even failures for RS data of various non-optical modalities. This has inspired growing interest in developing ==Remote Sensing Foundation Models (RSFMs)== to address the ==complex demands of Earth Observation (EO) tasks==, spanning the surface, atmosphere, and oceans. This survey systematically reviews the emerging field of RSFMs. It begins with an outline of their motivation and background, followed by an introduction of their foundational concepts. It then categorizes and reviews existing RSFM studies including their datasets and technical contributions across Visual Foundation Models (VFMs), [[VLM|Vision-Language Model]]s (VLMs), [[Large Language Model]]s (LLMs), and beyond. In addition, we benchmark these models against publicly available datasets, discuss existing challenges, and propose future research directions in this rapidly evolving field. A project associated with this survey has been built at [this https URL](https://github.com/xiaoaoran/awesome-RSFMs) .

# AI Summary
- Covers VFMs, VLMs, LLMs, and generative models tailored for EO
- Combines taxonomy, dataset review, and benchmarking across sensor modalities. Previous surveys were either narrower in scope or lacked empirical comparison.
- [[Segment Anything Model|SAM]] gets dedicated coverage, alongside vision transformers and autoregressive LLMs.
- Methods
	- Three paradigms in pretraining:
		- [[Supervised Learning]]: ImageNet or RS-specific datasets (MillionAID, SatlasPretrain)
		- [[Contrastive Loss|Contrastive Learning]]: Temporal, cross-model, geo-location pairs (SeCo, SkySense, CSP)
		- [[Masked Image Model|Masked Image Modeling]] (MIM): [[Masked Autoencoder]] (MAE) variants adapted for RS: Temporal embeddings (SatMAE), multi-scale (atMAE++), spectral masking (HyperSIGMA), multimodal (msGFM, MMEarth)
	- Fine-tuning/Adaptation: Full fine-tuning, LoRA/adapter-based [[Parameter-Efficient Fine-Tuning|PEFT]] (Dominant strategy), zero-shot prompting.
		- [[Segment Anything Model|SAM]] adaptions use frozen encoders + per-modality decoders.
- Key Experiments
	- Benchmarks: [[BigEarthNet]] (multi-label classification), [[EuroSAT]] ([[Land Use and Land Cover|LULC]]), [[SpaceNet]] (building segmentation), [[Functional Map of the World]] (fMoW) (time-series classification), [[Detection in Optical Remote Sensing Images|DIOR]] (object detection), [[GeoBench-VLM]] (31-task vlm eval)
	- Headline numbers: At the time, [[SkySense]] was the best... Contrastive-only methods (SeCo, CACo) lag behind MIM-based on detection tasks.
	- VLM zeros-hot models... GRAFT was able to outperform [[RemoteCLIP]] and CLIP-RCSID on [[EuroSAT]].
	- All open-source VLMs significantly underperform closed-source ones on GEOBench-VLM, all exhibit suboptimal geospatial reasoning.

Geospatial Specifics:
- What makes it geo-specific:
	- Multi-sensor modalities: [[Multispectral|MSI]], [[Synthetic Aperture Radar|SAR]], [[Hyperspectral|HSI]], [[Light Detection and Ranging|LiDAR]], [[Thermal Infrared|TIR]], [[Digital Surface Model|DSM]] (?), each with distinct spectral/geometric properties incompatible with standard FMs.
	- Top-down aerial perspective versus natural imaging perspective.
	- Temporal irregularity (multi-temporal change detection)
	- Geo-location metadata (lat/lon, [[Ground Sample Distance]], timestamps) as conditioning signals.
- Covered scene classification, semantic segmentation, object detection (horizontal and oriented), change detection, [[Visual Question-Answering]] (VQA)

Limitations:
- ==No [[Geospatial Embedding|RSFM]] shows "emergent capabilities", the dataset scale is insufficient==.
- Most work centers on optical RGB/[[Multispectral|MSI]]. ==Things like [[Synthetic Aperture Radar|SAR]], [[Hyperspectral|HSI]], [[Light Detection and Ranging|LiDAR]] coverage is thin.==
- ==Multimodal fusion gains== over single-modality fine-tuning are ==modest==; ==heterogenous sensor integration remains unsolved.==
- ==Zeor-shot generalization lags far behind general-domain FMs.==
- Model compression/edge deployment is nearly unexplored...
- Higher-order geospatial reasoning (causal, temporal, spatial relation inference) is essentially absent

Takeaway:
- RSFMs are technically active but still pre-paradigm-shift field. The architectures from natural vision transfers PARTIALLY but not fully!
- ==No RSFM has yet achieved the zero-shot generalization that makes general FMs transformative.==
- TAKEAWAY: ==The most pressing bottleneck is data==: More diverse, higher-resolution, multi-modal, temporally rich datasets with automatic annotation pipelines.



# Pretaraining mentioned
- [[Functional Map of the World|fMoW-RGB]]: 363.6k Optical RGB Satellite images across 62 land-use categories, sourced from [[Maxar]]'s [[QuickBird]], [[Maxar|GeoEye]]-1, and [[Maxar|WorldView]]-2/3 satellites.
- [[BigEarthNet]]: 1.2M [[Multispectral|MSI]] and [[Synthetic Aperture Radar|SAR]] images with 19 [[Land Use and Land Cover|LULC]] class labels, sourced from [[Sentinel|Sentinel-1]]/[[Sentinel|Sentinel-2]]
- [[Seasonal Contrast|SeCo]]: 1M unlabeled MSI images for [[Self-Supervised Learning|Self-Supervised]] learning, from [[Sentinel|Sentinel-2]] and [[National Agriculture Imagery Program|NAIP]]
- [[Functional Map of the World|fMoW-Sentinel]]: 822k unlabeled [[[Multispectral|MSI]] images, lower res version of [[Functional Map of the World|fMoW]]; [[Sentinel|Sentinel-2]] at 10m
- [[MillionAID]]: 1M RGB images with 51 LULC classes from multiple commercial satellites at 0.5-153m GSD
- [[GeoPile]]: 600k unlabeled RGB images from [[Sentinel|Sentinel-2]], [[National Agriculture Imagery Program|NAIP]], and others
- [[SSL4EO-S12]]: 3M unlabeled MSI + SAR images from Sentinel-1/2 at 10m
- [[SatlasPretrain]]: 856k tile images across RGB, MSI, and SAR with 137 class across 7 label types.
- [[MMEarth]]: 1.2M images spanning 12 paired modalities (RGB, MSI, SAR, DSM, etc.) from Sentinel-1/2, [[ASTER DEM]]
- [[HyperGlobal-450K]]: 450k [[Hyperspectral|HSI]] images from EO-1, Gf-5 satellites at 30m
- [[Hyper-Seg]]: 41.9k [[Hyperspectral|HSI]] images with segmentation masks from AVIRIS, [[Sentinel|Sentinel-2]], GF-Series


# [[Visual Question-Answering]] Datasets
- RSVQA-LR
- RSVQA-HR
- RSVQAxBEN
- RSIVQA
- HRVQA
- CDVQA
- FloodNet
- RescueNet-VQA
- EarthVQA

# Image-text Pre-training
- [[RemoteCLIP]]
- [[RS5M]]
- [[SkyScript]]

# Captioning
- RSICD
- UCM-Caption
- Sydney
- NWPU-Caption
- RSITMD
- RSICap
- ChatEarthNet

# Visual Grounding
- GeoVG
- DIOR-RSVG

# Multi-task Mixed:
- MMRS-1M
- GeoChat-Set
- LHRS-Align
- VRSBench
- TEOChatlas
- SARLANG-1M

# Downstream Benchmark DAtsets
- [[EuroSAT]]: LULC scene classification on Sentinel 2 imagery
- [[Onera Satellite]]: Change detection benchmark with bi-temporal satellite imagery.
- [[SpaceNet]]: Building footprint segmentation from high-res satellite imagery
- [[Detection in Optical Remote Sensing Images|DIOR]]-H and DIOR-R: Object detection benchmark; horizontal and oriented bounding boxes over 20 categories
- [[iNat2018]]: iNaturalist geo-tagged species images, used for geo-aware classification evaluation
- [[S2-100k]]: 100k Sentinel 2 image with global geographic coverage, used in SatCLIP
- [[GeoBench-VLM]]: 31 fine-grained tasks across 8 categories for evaluating VLMs on geospatial perception
- XLRS-Bench: Ultra-high-resolution RS imagery benchmark exposing limits of current models on very large images with small objects.



# Models mentioneD: Just hte best ones
VFMs (visual)
- [[SkySense]]: Top overall VFM, contrastive across RGB/MSI/SAR with geo-context prototypes
- [[msGFM]]: Best classification scores, multimodal MIM across RGB, MIS, SAR, DSM
- [[SatMAE]] and SatMAE++: Strong MAE baseline adapted for temporal. multi-spectral RS
- [[Scale-MAE]]: MAE with GSD-aware positional encoding for resolution variability
- [[HyperSIGMA]]: Leading model for hyperspectral imagery
SAM Adaptations
- [[CAT-SAM]]: Best few-shot SAM adaptation
- [[MM-SAM]]: Extends SAM to cross-modal inputs (RGB+ HSI, RGB + LIDAR, etc.)
VLMs:
- [[GRAFT]]: Bets zeros-shot RS classification/retrieval, aligns satellite with ground-level imagery
- [[GeoChat]]: Strong generalist RS VLM built on LLaVA, handles multiple tasks zero-shot
- [[EarthGPT]]: Extends to SAR + infrared modalities beyond RGB
- [[TEOChat]]: Best for temporal sequences, handles change detection conversations.
Specialized:
- [[GeoLLM]]: Best LLM adaption for geospatial prediction using OSM-enriched prompts
- [[DiffusionSat]]: Leading generating model for satellite image synthesis
- Pangu-Weather: Dominant weather-forecasting FM, outperforming traditional NWP systsems.








# Figures

![[Pasted image 20260422164948.png]]

![[Pasted image 20260422165009.png]]

![[Pasted image 20260422165014.png]]

![[Pasted image 20260422165023.png]]

![[Pasted image 20260422165033.png]]

![[Pasted image 20260422165041.png]]

![[Pasted image 20260422165059.png]]

![[Pasted image 20260422165107.png]]



![[Pasted image 20260422165116.png]]

![[Pasted image 20260422165124.png]]

![[Pasted image 20260422165134.png]]


![[Pasted image 20260422165158.png]]

![[Pasted image 20260422165212.png]]













