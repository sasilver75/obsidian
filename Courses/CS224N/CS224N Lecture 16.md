#lecture 
Link:https://www.youtube.com/watch?v=5vfIT5LOkR0

Subject: Multimodal Deep Learning

------
![[Pasted image 20240417000000.png]]

Today, Chris introduces us to [[Douwe Kiela]], the guest lecture for this lecture ([[Meta AI Research|FAIR]], where he invented [[Retrieval-Augmented Generation]], [[HuggingFace]]), now a Stanford Professor

------

Trying to keep things to what's useful

[[Multimodal|Multimodality]]: "Having, involving several modes, modalities, or maxima." In our case, focusing on NLP, it's *text plus one other modality (images, speech, audio, olfaction, others).* We'll mostly focus on images as the other modality.


## Why does MultiModality matter?
1. Faithfulness: Human experience itself is very multimodal
2. Practical: The internet and many applications is multimodal
3. Data efficiency and availability
	- We're sort of running out of data on the text side; to continue scaling, we need more tokens from other modalities!
	- Multimodal data is rich and "high bandwidth"
	- [[Yann LeCun]]: "\[Language is\] An imperfect, incomplete, low-bandwidth serialization protocol for the data structures we call thoughts."

![[Pasted image 20240417002242.png]]
Above: There are many ways to do intermodal fusion!

![[Pasted image 20240417002349.png]]
1. Combine modalities at beginning ("Early fusion")
2. Treat them separately and then later combine them ("Middle fusion")
3. Combine just the scores/logits ("Late fusion")
	- This is what we would now call contrastive models, eg [[CLIP]]

![[Pasted image 20240417002535.png]]
Just want to make sure that we rank the blue things higher than all otehr alternatives; nothing special about this architecture that was invented here, but it was cool because it was transformers, and transformers all the way (text: transformer, images: ViT) -- Trained on lots and lots of data.
- Alec Radford is a genius at creating large and high quality datasets.

==The speaker (Douwe Kiela) thinks that the CILP paper is one of the best papers ever written in our field -- extremely thorough and worth a read!==


![[Pasted image 20240417002825.png]]
Quickly after CLIP, a paper from Google called ALIGN, which really did the same thing and threw more images/text at it. 1.8B pairs instead of 300M, got a better model. Google.

An organization called [[LAION]] has started an open-source collective to create high-quality image-text pairs! (eg LAION 5B)
- Used to train [[Stable Diffusion]]

(Pick back up at 28:00)







