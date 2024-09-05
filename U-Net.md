---
aliases:
  - UNet
  - Unet
---
May 18, 2015
Paper: [U-Net: Convolutional Networks for Biomedical lmage Segmentation](https://arxiv.org/abs/1505.04597)
(It's interesting that these were actually developed for medical applications, but I'm most familiar in their (or a modified version of them) use in Diffusion models!)

Abstract
> There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at [this http URL](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net) .


![[Pasted image 20240418013639.png]]