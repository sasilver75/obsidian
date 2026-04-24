---
aliases:
  - Geo-Embedding
  - Geospatial Foundation Model
  - GFM
  - Remote Sensing Foundation Model
  - RSFM
  - Earth Observation Foundation Model
  - EOFM
---
References:
- [GeoEmbeddings best practices](https://geoembeddings.org/bestpractices.html)
- [CNG Geo-Embeddings Sprint](https://cloudnativegeo.org/blog/2026/04/geo-embeddings-sprint-march-2026/)
- [awesome-geospatial-foundation-models](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models)
- Video: [EO Geoawesome Foundation Models](https://www.youtube.com/watch?v=V3bHjyBp2-s)

See: [[AlphaEarth Foundations|AlphaEarth]], [[OlmoEarth]], [[Clay]], [[TESSERA]], [[Prithvi]], [[Prithvi v2]]

Instead of training a model from scratch for each [[Remote Sensing|Earth Observation]] task, pretrain a large model on massive satellite imagery datasets to learn general representations of Earth's surface, then fine-tune or use those embeddings for downstream tasks.
- A form of compression of an image or time-series, compressing it to a semantically-meaningful vector.

The question is: What [[Pretext Task]](s) gives us useful signal for a wide variety of tasks?



![[Pasted image 20260419232046.png]]
![[Pasted image 20260420005612.png]]
Above: [Source](https://www.youtube.com/watch?v=HPlZbkNa6zI) (2026)

![[Pasted image 20260420104753.png]]
In Isaac Corley's experiments, he was able to binary-encode these float32/64 vectors and saw that it still works, and then just used Hamming distance. Interesting! The PopCount algorithm is a fast way of comparing these Hamming distances. This lets you run this in the browser to do this retrieval without a cluster; everything is in cloud storage.

![[Pasted image 20260420105118.png]]
Claude is really good at building web applications and presentations! The [Terrabit](https://isaac.earth/terrabit/) app and these slides are all done in JavaSCript from a markdown. Domain expertise is a plus, agents is a force multiplier. Being able to prompt Claude from his phone and review it has been a game changer; you still need to know what success criteria looks like, etc.


![[Pasted image 20260420105612.png]]
Interesting! Research Template repository; Github integration to overleaf, etc. Free and public.

![[Pasted image 20260420110101.png]]
	


____________

Notes for [EO Geoawesome Foundation Models](https://www.youtube.com/watch?v=V3bHjyBp2-s)


![[Pasted image 20260424022654.png]]
Have multiple sensor modalities and imaging platforms within
Time is super essential for e.g. agricultural applications
For resolution, it's not just geospatial resolution, it's also spectral resolution (band count; hyper might have 200+ bands), and temporal (revisit time)


Talking on Geospatial embeddings...
![[Pasted image 20260424022839.png]]
- There's a paper he loves that use spherical harmonics function that embed places ont he sphere, on the globe, in a way where there's no edges (e.g. 180 to -180, pole problems with lat/lon)... you can go really deep in any of these dimensions... if you have a picture of a foxx, the fox might look really similar, but if you know where it lives, you have a higher chance of guessing what type of fox it is.


![[Pasted image 20260424022952.png]]
A growing number of models
- [[TESSERA]] is an excliting new type of dimension, focusing on pixels nad time, but there area lot of other dimensions that may or may not be included ina a full foundatino model hwere you try to improve evreything...
- THe foundation model space can get really broad. I would focus on ease of use first... Tessera has easy ot use SDKs... and then model capabilities ar obbviously dealbreakers... and obviously the size of the model and the licenses.


Now talking about [[Clay]], since hte spaker was involved in building it

![[Pasted image 20260424023425.png]]
We tried to make it flexible in the the:
- size
- resolution
- band count of images that are passed in
Used [[Segment Anything Model|SAM]] as a teacher network.. and had a [[Masked Autoencoder]] loss as a training goal.
![[Pasted image 20260424023518.png]]
For data, sampled using [[WorldCover]] as a reference, tryning to have an even distirbution, but oversampling in human footprint and agricultural areas.

Collected 70M chips from 6 different systems, all the way from drone to MODIS
![[Pasted image 20260424023601.png]]
See from sub meter resolution to 500.

![[Pasted image 20260424023619.png]]
Huge access to compute, lots of learning for the team...
- Did [[Data Parallelism|Data Parallel]] training, distributign the adta across nodes, but the model still fit on one GPU
- This gets a lot more complicated when you have models htat dont' fit on one gpu!

Here, they struggled in keeping these GPUs busy
![[Pasted image 20260424023706.png]]
They had the most scalable, most throughput system that you can get, and even that wasn't fast enough in our first aproaches, so in the beginning they were trying to.... put the chips sin dindividual tiff files, that failed, tried a bunch of things.. ended up bundling an entire batch into a single zip file so the datafile would juts load one zipfile per batch
That's diferent from iference; for infrence, go back to [[SpatioTemporal Asset Catalog|STAC]] and do direct extraction from larger files as you need them. 


Some downstream tasks for testing clay (not as thorogouh as [[TESSERA]] for instance)
![[Pasted image 20260424023845.png]]


Now let's tlak about Embeddings, which are really hte talk of the day, and are very important.
- ==Data eats models for lunch==

The more models are going to be release and the better they get, the more they will be a bit comparable... for their application, they'll get pretty good models... whether or not oyou can use th specific embeddings depens mcuh more on the data than the model themselves.
- See TESSERA for agriculture....


Smooth over artifacts
![[Pasted image 20260424024111.png]]

![[Pasted image 20260424024008.png]]
On the left, most is just bare soil... but people wnat to tknow what type of crop it is! Without time eseries, u cant extract it.

If you're zoomed too much into a lake, you dont know if you're in the apcific, in lake como, lake baikal, etc.
![[Pasted image 20260424024050.png]]


![[Pasted image 20260424024133.png]]
If oyu want ot use a foundation model, should you use pre-ran embeddings or run your own? First, try to run on existing embeddings, but be aware that they might not get upadtes as frequently as you ned them, might not be supporte dlong term, and might not be practicala depending on what yo u need.


![[Pasted image 20260424024343.png]]
Integration of language... more and mre papers try to solve for this. You need image and text pairs for CLAY... a test using open street maps. This is something where.. there's some active resercht that needs to be done. 

![[Pasted image 20260424024420.png]]
Can think about other multimodal models that can take any input/output in these different spaces.


![[Pasted image 20260424024441.png]]
There are some benchmarks... There is still active discussio as to how to evaluate these models.

You can't attend a talk nowadays without talking aobut LLMs
![[Pasted image 20260424024518.png]]


![[Pasted image 20260424024752.png]]
At DevSed we've been lucky... but this will probably get less and less ince not everyone will build their own FMs. Some colleageus say stuff is boring because they don't do their own model development anymore. They do some finetunnig, maybe.

![[Pasted image 20260424024917.png]]











