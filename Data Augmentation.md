A technique used in ML to increase the size and diversity of the training dataset by creating modified versions (eg translated, rephrased, rotated, fuzzed, inverted) of the training data. Increases the robustness of models, assuming that the modified data reflects possible realities and still results in the same label.

Pros:
- Valuable when ground-truth data is limited, imbalanced, or lacks diversity. Data augmentation can make training data more representative and varied, thus helping to improve model generalizability and reduce overfitting.

Cons:
- Challenging to apply to tabular data (compared to image and text data). It can be tricky to ensure that synthetic data matches the distribution of actual data.

The classic example of data augmentation is in computer vision, where we augment images through geometric transformations like cropping, rotation, flipping, blurring, inversion, and grayscaling. These simple transforms increase the volume and diversity of training data, improving the performance of CV models while reducing the cost of data collection and labeling.

An example of data augmentation on text, in the context of DoorDash. We vary sentence order in food descriptions and randomly remove information from menu categories. This helps simulate real-world variation in menus, where merchants don't always have detailed descriptions or menu categories. When training their models, they used a ratio of 100 synthetic labels to 1 actual label.

![[Pasted image 20240525163539.png]]

Beside data augmentation, another way to expand the dataset is by creating purely [[Synthetic Data]] (in my opinion, data augmentation is just a subset of synthetic data where the generation is conditioned on actual data). 
- Cloudflare used synthetic data to increase the diversity of their training data that is used to train models to classify malicious HTTP requests. They created negative samples using pseudo-random strings based on a probability distribution of existing tokens in the training data. The goal was to desensitize the model to individual tokens and keywords, and have it focus on the higher-level structural and semantic aspects of the payload. This reduced the false positive rate by ~80%.
- Meta used synthetic data generators to create sensitive data such as social security numbers, credit card numbers, addresses, etc. 
- Uber also generated synthetic data based on their experiences with and assumptions of fraud attacks, to validate their anomaly detection algorithms during automated testing of data pipelines. Libraries like Faker make it easy to generate fake names, addresses, phone numbers, and more.