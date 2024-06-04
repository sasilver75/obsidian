---
aliases:
  - TTA
---

Similar to what [[Data Augmentation]] does to the training set, the purpose of TTA is to perform modifications to the image we're doing inference on at test time.

We show our model various augmented versions of the image, and then average the predictions of each corresponding image, and take that as our final guess.