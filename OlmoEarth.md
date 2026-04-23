November 17, 2025
[Arxiv Link](https://arxiv.org/abs/2511.13655)
[Website](https://allenai.org/olmoearth)
[Technical Blog](https://allenai.org/blog/olmoearth-models)
[Github link: Data, training, and eval](https://github.com/allenai/olmoearth_pretrain)

Takeaway: Seems to indicate that you can build an [[Geospatial Embedding|GFM]] that's actually usable by organizations who need it.
Improvements in:
1. Training Stability: Other latent-space [[Self-Supervised Learning|SSL]] methods produce good features, but keep collapsing; their fix (frozen random projections as target encoder) is simple and works well ([[Latent MIM Lite]])
2. Honest Evaluation: No standard benchmark suite exists for EO models, so they pulled 12+ competitor models into a single evaluation framework with the same training recipe, which itself is a meaningful contribution.
3. Real-world deployment (actually used by nonprofits; tries to close the gap with users through an accessible platform, open model).

______


Abstract:
> ==Earth observation data presents a unique challenge==: it is ==spatial like images==, ==sequential like video== or text, and ==highly multimodal==. We present ==OlmoEarth==: a spatio-temporal, multimodal foundation model that employs a ==novel self-supervised learning formulation==, ==masking strategy==, and ==loss== all designed for the Earth observation domain. OlmoEarth achieves ==state-of-the-art performance compared to 12 other foundation models== across a variety of research benchmarks and real-world tasks from external partners. When evaluating embeddings OlmoEarth achieves the best performance on 15 out of 24 tasks, and ==with full fine-tuning it is the best on 19 of 29 tasks==. We deploy OlmoEarth as the backbone of an end-to-end platform for data collection, labeling, training, and inference of Earth observation models. The OlmoEarth Platform puts frontier foundation models and powerful data management tools into the hands of non-profits and NGOs working to solve the world’s biggest problems. OlmoEarth source code, training data, and pre-trained weights are available at https: //github.com/allenai/olmoearth_pretrain.


# Notes from AI Summary
- A family of [[Vision Transformer]]-based [[Remote Sensing|EO]] [[Foundation Model]]s in four sizes:
	- Nano (~1.4M params)
	- Tiny (~6.2M)
	- Base (~90M)
	- Large (~300M)
- Designed to be accessible to non-profit, humanitarian, and environmental organizations via an open end-to-end platform.
- Key contribution: [[Latent MIM Lite]]
	- A stable, [[Self-Supervised Learning|Self-Supervised]] training method that sites between the following:
		- [[Masked Autoencoder|MAE]]: In pixel space, stable but weak features.
		- [[Latent MIM]]: In feature space, better features but unstable/collapses.
	- Uses a randomly-initialized, frozen linear projection as the target encoder, avoiding collapse, maintaining latent-space benefits.
	- Unifies supervised and self-supervised learning: labeled maps and observations both go through the same frozen projection, same loss.
	- Modality-aware masking: Some modalities are encode-only, some are decode-only, makes the task harder without extreme masking ratios.
	- Modality Patch Discrimination loss: Contrastive loss only within the same bandset (avoids easy cross-modality negatives)
	- Instance Contrastive loss: Two random masking views of the same input, contrastive loss over pooled global embeddings (a la [[SimCLR]]).
- Data
	- 285,288 globally sampled locations, 2.56x2.56km, one year of data per sample
	- Observations (direct physical measurements, always ground truth):
		- [[Sentinel|Sentinel-1]]
		- [[Sentinel|Sentinel-2]] (3 bandsets)
		- [[Landsat|Landsat-8]] (2 bandsets)
		- NOT
			- "*We experimented with [[National Agriculture Imagery Program|NAIP]] (2.5m/p) and [[ERA5]] (160m/p), but ultimately dropped them from pretraining, finding no significant improvement on our evaluations.*"
	- Maps (human-derived labels; maps change over time)
		- [[WorldCereal]]: Crop type maps
		- [[WorldCover]]: Land cover maps
		- [[OpenStreetMap]]: Vector map features
		- [[Shuttle Radar Topography Mission|SRTM]]: Elevation/terrain
		- [[Cropland Data Layer]] (CDL): US cropland
		- Canopy Height Map (one from [[Global Ecosystem Dynamics Investigation|GEDI]], [[Sentinel|Sentinel-2]]): Tree canopy height
	- Locations chosen by sampling 120 [[OpenStreetMap]] feature categories.
	- During pretraining, OlmoEarth trains on both satellite imagery and on labeled maps. The maps serve as a supervision signal; the model learns to predict what the map labels should look like for a given location, which forces it to develop semantically meaningful representations.
	- But maps are never fed as input to the encoder; they only appear as targets for the decoder to predict. The encoder only ever sees satellite observations (Sentinel-1/2, Landsat)
	- At inference time, you ... just don't have current maps: That's usually hte whole point, you're trying to PRODUCE a map or classification, not consume one, and even if you DID have an old map, it might be outdated (land cover changes, new buildings, deforestation, etc.), so relying on it as an input could actively hurt you.
	- Design is:
		- Training: Encoder sees an observation, decoder predicts both masked observations AND map labels
		- Inference: Encoder sees operations: Attach a task-specific head and fine-tune/probe from there.
	- The maps give the model rich semantic grounding during training without creating a dependency on them at inference time.
- Architecture
	- [[FlexiViT]]-style patch embedding with  variable patch size (resizes input rather than changing projection weights).
	- 2D sincos positional + sinusoidal temporal + learnable modality embeddings.
	- Encoder: full self-attention across space, time, and modalities
	- Decoder: cross-attends masked tokens to visible encoder outputs; depth = 4 for all sizes
	- Maps are never encoded at inference (only used as training targets)
Evaluations:
- Compared against 12+ models:
	- EO Foundation Models
		- [[Galileo (Model)]], [[Panopticon]], [[TerraMind]], [[CROMA]], [[Clay]], [[Anysat]], [[Presto (Model)]], [[Prithvi v2]], [[Satlas]], [[TESSERA]]
	- General Vision Models applied to EO
		- [[DINOv3]], [[DINOv3 Sat]]
- kNN/Linear probing: best on 15/24 tasks
- Fine-tuning: best on 19/29 tasks

Datasets:
- Pretraining Sources
	- Sentinel-1 — SAR satellite imagery
	  - Sentinel-2 — multispectral optical imagery (3 bandsets)
	  - Landsat-8 — multispectral optical imagery (2 bandsets)
	  - WorldCereal — crop type maps
	  - WorldCover — land cover maps
	  - OpenStreetMap — vector map features (also used for sampling pretraining locations)
	  - Cropland Data Layer (CDL) — US cropland maps
	  - SRTM — elevation/terrain data
	  - Canopy Height Map — tree canopy height
	  - NAIP — high-res aerial imagery (tested, no improvement)
	  - ERA5 — climate reanalysis data (tested, no improvement)
- Research Benchmarks (from GEO-Bench unless noted)
	- m-bigearthnet — multilabel land use classification, S2
	  - m-so2sat — urban land use classification, S2
	  - m-brick-kiln — industrial site detection, S2
	  - m-forestnet — forest loss classification, Landsat
	  - m-brick-kiln — industrial site detection, S2
	  - m-forestnet — forest loss classification, Landsat
	  - m-eurosat — land use classification, S2
	  - m-cashewplant — crop detection, S2
	  - m-SA-crop-type — crop type classification, S2
	  - BreizhCrops — crop type time series, S2
	  - CropHarvest (Togo, PRC variants) — cropland detection, S1/S2
	  - PASTIS — agricultural parcel segmentation, S1/S2
	  - MADOS — marine debris segmentation, S2
	  - Sen1Floods11 — flood segmentation, S1
	  - m-eurosat — land use classification, S2
	  - m-cashewplant — crop detection, S2
	  - m-SA-crop-type — crop type classification, S2
	  - BreizhCrops — crop type time series, S2
	  - CropHarvest (Togo, PRC variants) — cropland detection, S1/S2
	  - PASTIS — agricultural parcel segmentation, S1/S2
	  - MADOS — marine debris segmentation, S2
	  - Sen1Floods11 — flood segmentation, S1
- Partner/Real-World Tasks
	  - AWF (African Wildlife Foundation) — LULC mapping, southern Kenya
	  - Nandi — (partner task, S1/S2/Landsat)
	  - GEANorthAfrica — time series classification
	  - ForestLossDriver — forest loss cause classification
	  - WorldCover — land cover maps
	  - OpenStreetMap — vector map features (also used for sampling pretraining locations)
	  - Cropland Data Layer (CDL) — US cropland maps
	  - SRTM — elevation/terrain data
	  - Canopy Height Map — tree canopy height
	  - NAIP — high-res aerial imagery (tested, no improvement)
	  - ERA5 — climate reanalysis data (tested, no improvement)
- Research Benchmarks (from GEO-Bench unless noted)
	  - m-bigearthnet — multilabel land use classification, S2
	  - m-so2sat — urban land use classification, S2
	  - m-brick-kiln — industrial site detection, S2
	  - m-forestnet — forest loss classification, Landsat
	  - m-eurosat — land use classification, S2
	  - m-cashewplant — crop detection, S2
	  - m-SA-crop-type — crop type classification, S2
	  - BreizhCrops — crop type time series, S2
	  - CropHarvest (Togo, PRC variants) — cropland detection, S1/S2
	  - PASTIS — agricultural parcel segmentation, S1/S2
	  - MADOS — marine debris segmentation, S2
	  - Sen1Floods11 — flood segmentation, S1
	
	  - NAIP — high-res aerial imagery (tested, no improvement)
	  - ERA5 — climate reanalysis data (tested, no improvement)
  Partner/Real-World Tasks
  - AWF (African Wildlife Foundation) — LULC mapping, southern Kenya
  - Nandi — (partner task, S1/S2/Landsat)
  - GEANorthAfrica — time series classification
  - ForestLossDriver — forest loss cause classification
  - LiveFuelMoistureContent — regression task
  - Mangrove — detection/segmentation
  - MarineInfrastructure — detection
  - VesselDetection / VesselLength / VesselType — maritime tasks
  - SolarFarmDetection — infrastructure detection



# Notes from Reading
- 






# Figures

![[Pasted image 20260422182752.png]]



![[Pasted image 20260422182804.png]]


![[Pasted image 20260422182823.png]]

![[Pasted image 20260422182829.png]]

![[Pasted image 20260422182841.png]]


![[Pasted image 20260422182848.png]]

![[Pasted image 20260422182857.png]]


![[Pasted image 20260422182904.png]]

![[Pasted image 20260422182913.png]]



# Notes from Appendices
- 





