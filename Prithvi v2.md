Dec 2024, [[IBM]] + [[National Aeronautics and Space Administration|NASA]]
Paper: [Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications](https://arxiv.org/pdf/2412.02732)

Uses a 3D [[Masked Autoencoder]], extending the standard [[Vision Transformer|ViT]] MAE to spatiotemporal cubes.
- 3D path embedding via [[Conv3D]], treating (time x height x width) as a cube.
- 3D positional encodings
- Metadata embeddings (latitude, longitude, year, day of year encoded and added to tokens)
- Metadata [[Dropout]] during pretraining (p=0.1), model learns to work with or without location/time information.


![[Pasted image 20260419155715.png]]






