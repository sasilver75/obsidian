# Topic: Multimodal Alignment
https://www.youtube.com/watch?v=iSz0WAzyFHE&list=PL-Fhd_vrvisMYs8A5j7sj8YW1wHhoJSmW&index=9

---

![[Pasted image 20240618153529.png]]
==Connection==: Knowledge from one modality that provides information about the other modality. In statistics, this is also known as *==association==* (eg between features) or ==dependency==. It means that when we say "Kitchen," there's some object that you expect in the image, or if you see an image, there are certain words you expect.

![[Pasted image 20240618154758.png]]

We did an example of CLIP
- Where the InfoNCE models something very close to mutual information
![[Pasted image 20240618155026.png]]
Mutual Information... at the basics from information theory, if you had to associate one emotion with information theory, it's *surprise* -- if something is surprising in data (eg a flat distribution), that's low information; something peaky has high entropy/information -- that's why information is 1/p(x).

We mathematically designed this Mutual INformation equation...
- "The probability of th two event happening jointly... divided by the probabilities of the events happening themselves"
- The mutual information formula is the ratio between them, or it could be the (KL) distance.
	- The numerator... is like... "joint event"..."co-occurrence". 

Modality Interactions
- There's no interaction without a response -- it *needs*a response. Interactions happen during inference (from a human or a model).
	- A human: "I'm seeing those modalities, and I'm inferring"...
![[Pasted image 20240618155653.png|300]]
![[Pasted image 20240618160121.png|300]]
Mutual Information really just means co-occurrence -- just joint probability (normalized by products of marginal probabilities).
![[Pasted image 20240618161956.png|300]]
Three-way mutual information is kind of undefined... there's a lot of argument about how to compute this.
- Sometimes when you mathematically look at it, it becomes negative 🤔 Probabilities are always positive, usually!





Anyways, today we'll talk more about Modality Alignment





