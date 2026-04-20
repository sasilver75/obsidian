---
aliases:
  - Geo-Embedding
  - Geospatial Foundation Model
  - GFM
  - Remote Sensing Foundation Model
  - RSFM
---
References:
- [GeoEmbeddings best practices](https://geoembeddings.org/bestpractices.html)
- [CNG Geo-Embeddings Sprint](https://cloudnativegeo.org/blog/2026/04/geo-embeddings-sprint-march-2026/)
- [awesome-geospatial-foundation-models](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models)

See: [[AlphaEarth Foundations|AlphaEarth]], [[OlmoEarth]], [[Clay]], [[TESSERA]], [[Prithvi]]

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
	
