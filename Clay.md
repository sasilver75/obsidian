May, 2024
An open-weight [[Geospatial Embedding]]/[[Geospatial Embedding|Geospatial Foundation Model]] built by [[Development Seed]] and the Clay Foundation. 


References:
- [Video: Clay AI for Earth. Technical from Zero to Hero](https://www.youtube.com/live/Zd3rbBj56P0?si=H3pFcFcx_5bpxXhU)

Clay is a foundational model of Earth. It uses an expanded visual transformer upgraded to understand geospatial and temporal relations on Earth data. The model is trained as a [[Self-Supervised Learning|Self-Supervised]] Masked Autoencoder ([[Masked Autoencoder|MAE]]).

==It doesn't seem like they've released a paper==. Built by the Clay Foundation, within the context of [[Renaissance Philanthropy]].
- Bruno Sanchez-Andrade Nuño: Exec. Director of Clay and former NASA astrophysicist
- Dr. Dan Hammer: Co-founder of Clay, also a fellow at Renaissance Philanthropy and co-founder of LGND AI.

Use cases:
- Generate semantic embeddings for any location and time. You can use embeddings for a variety of tasks, including to:
	- Find features: Locate objects or features, such as surface mines, aquaculture, or concentrated animal feeding operations.

- Fine-tune the model for downstream tasks such as classification, regression, and generative tasks. Fine-tuning the model takes advantage of its pre-training to more efficiently classify types, predict values, or detect change than from-scratch methods. Embeddings can also be used to do the following, which require fine-tuning:
	- Classify types or predict values of interest: Identify the types or classes of a given feature, such as crop type or land cover, or predict values of variables of interest, such as above ground biomass or agricultural productivity.
	- Detect changes over time: Find areas that have experienced changes such as deforestation, wildfires, destruction from human conflict, flooding, or urban development.
	- This can be done by training a downstream model to take embeddings as input and output predicted classes/values. This could also include fine-tuning model weights to update the embeddings themselves.

- Use the model as a backbone for other models.




Trained on multi-spectral, multi-temporal satellite imagery ([[Sentinel|Sentinel-2]], [[Landsat]], [[National Agriculture Imagery Program|NAIP]])

