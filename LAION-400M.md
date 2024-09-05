August 20, 2021 -- [[EleutherAI]]
Paper: [LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-tExt Pairs]()
Blogpost: [LAION-400-MILLION OPEN DATASET](https://laion.ai/blog/laion-400-open-dataset/)

A dataset of 400M image-text pairs, along with their CLIP embeddings an kNN indices, allowing efficient similarity search. Used to train multi-modal [[VLM|Vision-Language Model]].

We have filtered all images and texts in the LAION-400M dataset with OpenAI‘s [CLIP](https://openai.com/blog/clip/) by calculating the cosine similarity between the text and image embeddings and dropping those with a similarity below 0.3. The threshold of 0.3 had been determined through human evaluations and seemed to be a good heuristic for estimating semantic image-text-content matching.

Abstract
> Multi-modal language-vision models trained on hundreds of millions of image-text pairs (e.g. [[CLIP]], [[DALL-E]]) gained a recent surge, showing remarkable capability to perform zero- or few-shot learning and transfer even in absence of per-sample labels on target image data. Despite this trend, to date there has been no publicly available datasets of sufficient scale for training such models from scratch. To address this issue, in a community effort we build and release for public LAION-400M, ==a dataset with CLIP-filtered 400 million image-text pairs, their CLIP embeddings and kNN indices that allow efficient similarity search==.


> **==WARNING==**: be aware that this large-scale dataset is non-curated. It was built for research purposes to enable testing model training on larger scale for broad researcher and other interested communities, and is **not** meant for any real-world production or application.



