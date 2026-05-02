A more secure replacement for `pickle` serialization, developed specifically for use in sharing tensors related to machine learning, developed and popularized by [[HuggingFace]] in 2021 when they created it and made it the default format for model weights on the Hub.


On the inside, looks something like this:
![[Pasted image 20260501151742.png]]
Essentially a large JSON file containing the models' weights


