June 25, 2025 (~3m before [[AlphaEarth Foundations|AEF]])
[Arxiv Link](https://arxiv.org/abs/2506.20380)

Name is soemthing like: "Temporal Embeddings for Earth Representations and Analysis"

References:
- Video: [Madeline Lisaius on Geoawesome re: TESSERA](https://www.youtube.com/watch?v=u3_HxRJE8Ss)
- Video: [Robin Cole Interview](https://www.youtube.com/watch?v=10CBuGfrz6M&t=1s)


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


____________________

Video: [Robin Cole Interview](https://www.youtube.com/watch?v=10CBuGfrz6M&t=1s) January 20, 2026

Q: What's temporal about this model?
A: Tempoeral in the sense that it's not a point in time snapshot, they're ==annual embeddings that give you the temporal spectral characteristics of that point of earth over one calendar year.== 

Q: Waht about in practical terms in terms of the data that's available, etc?
A: For things like plants, the phenology, how the spectral chracteristics of those platns and how they change over the year are very distinctive. By capturing it over a year, we're able to discriminate over things which... if you just took a point-in-time snapshot from Sentinel-2, you couldn't discriminate between two plants that are very close by.
A: For training, it means that we need a lot of data. to produce embeddings for year, ew need an entire year's of data for that particular point. We need todownload all of the [[Sentinel|Sentinel-1]] and [[Sentinel|Sentinel-2]] data. Since it's a temporal model, we also pull down the whole year of data. And now we're doing embeddings from 2017 to 2025, so we need to pull down all the available Sentinel1 and Sentinel2 data; so it's something like 25 Petabytes of data, so it's a massive, huge amount of data.

Q: Do you train on all of it or curate?
A: We select... in training... over 3,000 MGRX(?) tiles... across the globe. We have deserts, rainforests, etc...
Q: Does it involve a lot of work, choosing those locations?"
A: It's kind of straightforward. We elect randomly, but we have to keep in mind that e have to avoid choosing some times that have overlaps with our validation data. We have a crop dataset in our validation dataset pipeline, so when choosing the ata, we hve to exclude the tiles that fall into that region of interest.

Q: It's worth pointing out.... that Tessera is unuusal in model training... Usually hte hard part is trianing and inference is relatively cheap; in Tessera's case, it's the other way around. Training requiers a lot of work and data movement, but inference is very expensive, because then we have to havae lal of the dat we need to produce those embeddings across all of those years, across both modalities. 

Q: What's the data requirement for a single embedding?
A: On average something like 76 Sentinel 2 observations, and for Sentinel 1 we have more data in Europe, less in Africa... Perhaps 60 observations.
A: About 180,000 GPU hours for the 7 years of embeddings across the globe?

Q: In practice, you're not expecting people to compute embeddings you've just done it and distributed them, is that right?
A: Yes

Q: What decisions did you have to make when choosing training strategies?
A: Tessera is a pixel-based model, so it's quite different from other patch-based foundation models. In termso ftraining strategy, we ... have several ways to train the model. 
- If we only select all the pixels in a patch, we mean that most of hte pixels look very similar ina patch
- So during the trianing we actuallly dgather as many pixels as possible within the globe... and then randomly sample them, basically. This makes sure that .... the loss convergence is smooth, and prevents feature collapse. 
- There's some other stuff we do.... we set up a very large batch size... I thin something like over 32,000 in temrs of batch size. There are some other tricks we do in terms of optimizer... we do learning rate warmup ([[Learning Rate Schedule]] ) and [[Cosine Annealing]]


Q: You mentioend it's pixel-based, there are otehr images that are image based... how does it differ for  pixel-based?
A: we adopted a different way for self-supervised training; lots of patchbased methods use [[Masked Autoencoder]] for [[Self-Supervised Learning|SSL]] training. We can simply mask along the temporal dimension for ipxels, rather than the spatial dimension; you mask some observations and predit int he spatial one... but we do something quite diffeenrt, we use [[Barlow Twins]], wehre we sample two observations from a year of observations, and we want to make sure taht althoguh htese two subset observations are different... we want to reduce the reduction (?) within these two subsets whil;e maintaining the essential information.
This is the first foundation model that uses this =="redundancy reduction method" of Barlow Twins==

Q: Intuition being that it should reconstruct similar embeddings if it has the second vs first half of the year?
A: Not two parts of the yar, two sets of observations that span the year, but might be differnt;t given two sets of observations at the same point, you want the same representation. The observations being a subset of the total observations.
Q: So you randomly select  fraction for set And the remainder for set B. Frank is underselling the amount of architectural work and hyperparameters observation stuff. Getting Barlow twins to continue scaling is quite tricky without having the representation to collapse after a while

Q: Are those tricks you learned along het way?
A: We adopted some common practices... when training the model... I almost gave up the model training, it's quite a pain. At some point, we realized... some issues that we dove into and analyzed, and kept iterating ,etc... and after several times, the loss converges and the embedding gets better! We learned a lot along the way.


When training the moel we nated otm ake sure that the data used for evalution are not in the training set.
Training with earth data is quite different from training with language models... becaues in remote sensing... basically we have limited data, we have a lot of satellite images.... we just exlcude the region of interest that falls into the validation dataset.


q: Do different teams agree on the different evaluation tasks and datsets? Or is that something that's being figured out?
A: We had a couple of evaluation tasks that we had also reported during training. We don't just want our losses to converge and later find out that our loss being low had nothing to do with performance on downstream tasks. So we had a couple of tasks in the training loop where we had in there so we can see that the performance on downstream tasks was improving as we scaled parameters and scaled data.


![[Pasted image 20260424183500.png]]

![[Pasted image 20260424183510.png]]

![[Pasted image 20260424183540.png]]



Q: It seems like these models have solved the annual tasks at this point, is that controversial?
A: We don't know! ==The paper has a bunch of scaling work to show that the embeddings get better and our performance on downstream tasks improves. Maybe, but this is also as bad as they're going to get; they'll only get better from here.== 

Q: So what's the immediate impact of this new capability?
A: Instead of having to deal with clouds, or gaps in missing observations, etc.. it's easy to take Tessera and put a simple model on top... The model and weights are all open source, etc. You don't have to rely on us to use it!

Q: Will people use just the embeddings, or are the embeddings a stepping stone? 
==A: TESSERA should be your baseline. Performance might be exactly what you want.== It has another propety, which is that it's incredibly label-efficient. Because you're fitting a model only to the 128 dim embeddings, and not trying to learn those physical features (?), you're not spending those precious label fitting parameters that are more about the data distribution of the world... so you only need a tiny fraction of the labels.


Q: Closing up... on what's next on the research dimension, then. What are other avenues other than scaling?
A: Could do change detection, looking year-to-year ot understand how things changed, and what... because you've abstracted away... clouds, etc... we've got colleagues looking at change detection, forecasting (in the latent space)

Q: What about modalities? Do you tink different modalities will be part of future models?
A: Yes, this is something we're trying to do... for change detection.... one of my major tasks is developing ==Tessera V2==... we're going to add [[Landsat]] in (along with S1, S2)... and for Landsat we have data since 1970s! So we can generate embeddings from 1970s onwards, which will give you a very long time series of embeddings, which will open the doors for any other tasks. This is something that I've been doing and testing.
A: Also there's alternative [[Synthetic Aperture Radar|SAR]] data sources which will increase the temporal resolution. 
That paper is on arxiv and under peer reviwe ath te moment












