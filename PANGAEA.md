December 5, 2024
[Arxiv Link](https://arxiv.org/abs/2412.04204)



Abstract:
> [[Geospatial Embedding|Geospatial Foundation Model]] (==GFMs==) have emerged as powerful tools for extracting representations from Earth observation data, but their ==evaluation== remains inconsistent and narrow. ==Existing works== often evaluate on suboptimal downstream datasets and tasks, that are often ==too easy or too narrow==, limiting the usefulness of the evaluations to assess the real-world applicability of GFMs. Additionally, there is a distinct ==lack of diversity== in current evaluation protocols, which fail to account for the multiplicity of image resolutions, sensor types, and temporalities, which further complicates the assessment of GFM performance. In particular, ==most existing benchmarks are geographically biased== towards North America and Europe, questioning the global applicability of GFMs. To overcome these challenges, we introduce ==PANGAEA==, a standardized evaluation protocol that ==covers a diverse set of datasets, tasks, resolutions, sensor modalities, and temporalities==. It establishes a ==robust and widely applicable benchmark for GFMs==. We evaluate the most popular GFMs openly available on this benchmark and analyze their performance across several domains. In particular, we compare these models to supervised baselines (e.g. UNet and vanilla ViT), and assess their effectiveness when faced with limited labeled data. Our findings highlight the limitations of GFMs, under different scenarios, showing that they do not consistently outperform supervised models. PANGAEA is designed to be highly extensible, allowing for the seamless inclusion of new datasets, models, and tasks in future research. By releasing the evaluation code and benchmark, we aim to enable other researchers to replicate our experiments and build upon our work, fostering a more principled evaluation protocol for large pre-trained geospatial models. The code is available at [this https URL](https://github.com/VMarsocci/pangaea-bench).

![[Pasted image 20260419161002.png]]

![[Pasted image 20260419161020.png]]

