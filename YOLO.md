---
aliases:
  - You Only Look Once
---
June 8, 2015
Paper: [You Only Look Once: Unified Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

Many variants/versions of this model exist, it's been very successful.
At the time of writing, we're at YOLOv8 ([Family Summary Paper](https://arxiv.org/html/2304.00501v6))

![[Pasted image 20240418000723.png|350]]

Abstract (for v1)
>We present YOLO, a new approach to ==object detection==. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.
>Our unified architecture is ==extremely fast==. Our base YOLO model processes images in real-time at 45 frames per second. A smaller version of the network, Fast YOLO, processes an astounding 155 frames per second while still achieving double the mAP of other real-time detectors. Compared to state-of-the-art detection systems, YOLO makes more localization errors but is far less likely to predict false detections where nothing exists. Finally, YOLO learns very general representations of objects. It ==outperforms all other detection methods==, including DPM and R-CNN, ==by a wide margin== when generalizing from natural images to artwork on both the Picasso Dataset and the People-Art Dataset.