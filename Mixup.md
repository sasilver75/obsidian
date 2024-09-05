The data augmentation technique from 2018

See also: [[MixMatch]] (2019), a holistic approach to semi-supervised learning that incorporates MixUp.

![[Pasted image 20240701115059.png]]
You can actually (in RGBA).... combine multiple images... and we give the model ("60% cat, 40% dog") as a label, rather than just "cat".  You don't have to one-hot encode your labels (think: multiclass classification), and you can still do a cross-entropy loss on it.