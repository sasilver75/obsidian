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



