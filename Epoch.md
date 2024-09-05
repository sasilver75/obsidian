Resources:
- [Article: Does Batch Size Matter? (Jane Street)](https://blog.janestreet.com/does-batch-size-matter/)



Larger batch size usually means less noisy updates.
It's good practice to reshuffle your dataset at every epoch to reduce model overfitting.
To fill: How many Epochs is considered useful, when pre-training or fine-tuning LLMs? There are a number of papers on this.


> "In general, if you train for more epochs, 3 is generally the most you want to do. 1 is generally the best, assuming you have the data available to train on." - Daniel Han, July 2024 [link](https://youtu.be/pRM_P6UfdIc?si=G8yn2Ro4Gfo-qPD_&t=1001)

