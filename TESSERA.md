June 25, 2025 (~3m before [[AlphaEarth Foundations|AEF]])
[Arxiv Link](https://arxiv.org/abs/2506.20380)

References:
- Video: [Madeline Lisaius on Geoawesome re: TESSERA](https://www.youtube.com/watch?v=u3_HxRJE8Ss)


Abstract
> Satellite [[Remote Sensing|Earth Observation]] (EO) time series in the optical and microwave ranges of the electromagnetic spectrum are often irregular due to orbital patterns and cloud obstruction. [[Mosaic|Compositing]] addresses these issues but loses information with respect to vegetation phenology, which is critical for many downstream tasks. Instead, we present [[TESSERA]], a ==pixel-wise foundation model for multi-modal (Sentinel-1/2) EO time series that learns robust, label-efficient embeddings.== During model training, TESSERA uses ==Barlow Twins== and ==sparse random temporal sampling== to enforce invariance to the selection of valid observations. We employ two key regularizers: ==global shuffling== to decorrelate spatial neighborhoods and ==mix-based regulation== to improve invariance under extreme sparsity. We find that for diverse classification, segmentation, and regression tasks, ==TESSERA embeddings deliver state-of-the-art accuracy with high label efficiency==, often requiring only a small task head and minimal computation. To democratize access, adhere to FAIR - principles, and simplify use, we release global, annual, 10m, pixel-wise int8 embeddings together with ==open weights==/code and lightweight adaptation heads, thus providing practical tooling for large-scale retrieval and inference at planetary scale. All code and data are available at: [this https URL](https://github.com/ucam-eo/tessera).




______________________


Video: [Madeline Lisaius on Geoawesome re: TESSERA](https://www.youtube.com/watch?v=u3_HxRJE8Ss)

## Intro to Foundation Models
- Built on [[Self-Supervised Learning|SSL]], built on the premise that in many domains we have a huge amount of unlabeled data, e.g. RGB images on the internet. Functions by trying to sort all this data into a multidimensional space...
- To use these embeddings, we can use some labels to contextualize this multidimensional and make these embeddings useful downstream.
- We have lots of situations with limited labels in [[Remote Sensing|EO]], petabytes and petabytes of [[Landsat]], on just a single sensor; a perfect fit.
- So the input to our SSL is geospatial data; we put this into our SSL algorithm, and the model can output vector embeddings, the location of data in a multidimensional space.
- 

## Before TESSERA: ST-BT: Spectral-Temporal [[Barlow Twins]]
- Focused on agriculture, where we have questions to answer like... taking the cases of a few grasses (wheat, barley, rye) and many other domesticated grasses that all look very similar from space; it's hard to discriminate these, much less do downstream tasks like yield estimation, etc.
- To develop methods that would be more useful for the agricultural space... ==I was intersted in pixel-wise multipsectral time series data.==
	- An inherent belief that ==the temporal evolution of spectra is the most important component for understanding our agricultural landscpes!==
		- This contrasts with CV approaches that focuses on spatio-temporal information as patches, etc.
		- Intuition: Imagine large industrial fields in Brazil or the US, a soy field. If we go to the center of the field and consider a pixel or patch, and think of its neighbors; we';ll see the same crop managed in the same way, on the asme soil. All we get from looking at neighbors is "We're in a big field." 
		- If we look at a small-holder ag context, like an Indian wheat field (10m x 20m); if we look at neighbors, we'll probably get different crops and different managements... we learn that we're in a small holder context, have a lot of change, and are very discrete. But we know that we're doing that already, because we're in northern India...
	- So this is the background of why we're focused on pixel-wise approaches, looking at the different spectra, and also looking at time-series data.
![[Pasted image 20260424113743.png]]
We looked for architectures that would be compatible with this idea/world.
- Barlow Twins take in images one by one and create two distortions of the image.
- An encoder works to take the image and compress it into a low-dim vector, which is then expanded back out into embeddings in the Barlow twins model, which are each pushed to be the same by the loss function.
- The loss function works to widen the cross-correlation matrix between the two embeddings of the two distorted images... meant to reduce the amount of information tha the model is learning about the distortions, and make sure that the dimensions of the embeddings have very ow correlation, so that teach dimension of the embedding is as unique as possible, and has as much unique information about the original image as encoded as possible.
- So... if we think about this learning mechanism, and about the geospatial context... an easy reinterpretation: ==Treat satellite images like RGB images. We can take two patches of satellite imagery... And do something like a rotational augmentation, put it into Barlow twins, and force the model to understand these as the same.==
- ![[Pasted image 20260424114026.png]]
	- If we go back to our assumption ad belief that ==the pixel-level spectral and temporal information is the most important==, we'll see that this isn't really compatible with our beliefs about the world.
	- So how can we re-imagine this?
	- ![[Pasted image 20260424114111.png]]
	- We organize data in a "deep pixel" format, taking the time series of acertain region, and for each pixel, by timestep, extract the bands, and organize it as a two-dimensional array.
	- Now we have a 2D array that's compatible with many of the models that expect some sort of multi-dimensional input.
	- Now that we're interpreting hte dat apixel-wise as temporal spectral resolutoin.. we can come back to barlow twins.
	- ![[Pasted image 20260424114206.png]]
	- This is where our [[Spectral-Temporal Barlow Twins]] comes from! 
	- If we take the time-serise (say, for a year) and have multiple dates where there is cloud cover, To create our two "natural augmentations," we can take a random subsample of dates two separate times, orered.
	- So a set of random dates, and as second set of random dates, and consider these to be two views of the same pixel, and input that into the barlow twins architecture... the model then... learns and becomes invariant to the augmentations, becoming invariant to date availability, invariance to missing data due to cloud corruption... and so we build a model that can tolerate this, which is especially important in the tropics, but also all across the globe.
- To be specific about what we'ret alking about when doing downstream tasks... 
- ![[Pasted image 20260424114550.png]]
- When we tlak about downstream tasks, we use the representation, which is the compressed version (n=128dim) that we use
- This approahc compresses data quite significantly! If you have 70 timesteps in the year and 10 spectral bands, that's 700 dimensions of data. If you put this into barlow twins, you get a 128-dim vvector, corruption from clouds is handled, we don't have to worry about missing dates, and we found taht using ST-BT gave us equal or even better results than using raw data.
	- So we found 


## TESSERA
- ![[Pasted image 20260424114836.png]]
- What makes it different? the Assumptions are almost the safe, but instead of having just the twin pipeline... for the [[Sentinel|Sentinel-2]] multispectral data, we also have [[Sentinel|Sentinel-1]] ([[Synthetic Aperture Radar|SAR]]) data incorporated in a similar way!
- Data is taken as a ==deep pixel== once again to use as input into the encoder, and after the encoder step, the outputs from the Sentinel 1 and Sentinel 2 twin siblings are glued together, and the glued together version are put into the projector, and we have the same acting loss function. So this s just a way to incorporate a second data modality into the pipeline described.
- Otherwise, we made the encoder slightly more complex, and, after we did architectural changes, we had a vision to make [[TESSERA]] be a global foundation model, training on thousands of images from around the globe to craete annual embeddings of n=128dim for the entire global surface.
- ![[Pasted image 20260424115119.png]]
- This included a huge amount of infrastructure engineering work and challenges when it came to building out pipelines; this took a lot of time for us.
- ![[Pasted image 20260424115114.png]]
- Some intuition around TESSERA:
	- A visualization, just visualizing 3 dimensions of these embeddings; we already have a colorful (figuratively, literally) understanding of the landscape.
	- Already, we can see visually that a lot of information is encoded into the embeddings of TESSERA that we can use...
	- Let's discuss some of the findings we've seen so far.
- The first example: Embeddings of a 2020 California fire, using a [[UMAP]] visualization (unsupervised clustering):
- ![[Pasted image 20260424115235.png]]
- We're seeing two things in this case: Although the embeddings are for an entire year, ==we see that temporal information is still included; embeddings that relate to an august fire event are separate from those in a june fire event.== 
- We see that within these fire events, we see a  ==clear gradient between the low-burn areas and the high-burn areas, so this severity is inherently encoded.==
	- (Obviously this is super simplified because it's two dimensions)
- Another example:
	- ![[Pasted image 20260424115344.png]]
	- This is an unsupervised clustering of the data.. then colored with land covers.
		- ==We see a clear separation of land cover types==, naturally within the embeddings.
	- To give another example... this is TESSERA used as an input to do canopy height estimation in Borneo
		- ![[Pasted image 20260424115458.png]]
		- TESSERA used as an input to predict canopy height; see that it has quite good [[Coefficient of Determination|R-Squared]]
			- (GSE is google alpha earth)
	- Finally, for the most complex types, crop type mapping in Austria:
		- ![[Pasted image 20260424115556.png]]
		- When used as an input for crop type classification, still ==performs better than many of the other baselines for crop type mapping==.
	- WHY TESSERA?
	- ![[Pasted image 20260424115644.png]]
		- Like [[Spectral-Temporal Barlow Twins|STBT]], we're compressing a year of data into 128 dimensions
		- There's no feature engineering needed...
		- In American english, we'd say it's very "plug and play"
		- We need significantly less compute to do downstream tasks, because we're working with much smaller data.
		- Performance with TESSERA is matching or beating other methods.
		- We're working to precompute embeddings globally, via the GeoTessera Python package!

## What's Next
- We've got a paper on arxiv that describes the 
- We have our entire model architecture on Github at the TESSERA page.
- We have the precomputed embeddings available by the Python package Geo-Tessera
- ==IF YOU WANT TO JOIN US==, we're looking for $500,000 to generate embeddings from 2017-2026, as well as storage space
	- lol... kay.
- We're seeing that embeddings have huge potential


