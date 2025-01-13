See also: [[Contrastive Loss]] (which Triplet Loss improves on)
See also: [[InfoNCE]] (which can be seen as an improvement on this)

References: [RitvikMath: Triplet Loss](https://youtu.be/PfHtXVWVgsQ?si=ZMG9r-bWWEbcQs6W)

We said that [[Contrastive Loss]] is context-unaware, by looking at the examples of [[Hard Negative]] and [[Hard Positive]] and how Contrastive Loss deals with them.

Triplet introduces triplets of three datapoints:
- Anchor, e.g. a picture of our person
- Positive (A same-class example as the Anchor), e.g. the same person in different lighting
- Negative (A different-class example from the Anchor), e.g. a different person

The goal of Triplet Loss is similar to that of Contrastive Loss: =="Given an anchor, positive, and negative, the negative example should be at *least* $M$ further away from the Anchor than the positive example."==

![[Pasted image 20250112170958.png|400]]
Intuition: "I want my loss to prioritize separating examples of different classes by a margin of at least M." I want to make sure that no matter how far away a positive example is from me, I want to make sure that the negative example (Which definitely shouldn't be close to be) is at least M distance further away than the positive example is.

So the structure of the positive cluster might be very tight or might be very far away, depending on the class. We're trying to get an optimal structure of the embedding space.

So each training example is an (A, P, N) triplet, and the loss of the triplet is $L_T = \max{[0, d_{AP}-d_{AN}+m]}$ . In other words, it's how much closer the negative is from the {AP distance plus a margin}.

![[Pasted image 20250112171722.png|500]]
Let's do some cases:
- Let's say the differential distance is less that negative M. If we add M to that, we still have a negative number, and Max(0, NegativeNumber) is still 0. That's why the triplet loss is still 0. Why does that make sense? If the quantity on the X axis is very negative, then the negative example is AT LEAST M units of distance further from the anchor than the positive is from the anchor. That's what we want! Well-separated classes. So it makes sense that the loss is zero here.
- Let's say that the differential distance is between -m and 0. In this case, we know that the negative case is further from the anchor than the positive case, but the margin is less than M... so we want to give a little bit of loss in this case so that the model learns to push that negative even further away than it already is.
- The last case is anything positive. If the differential quantity is positive, our negative is closer to our anchor than our positive, and that's not good! So the loss is pretty high here to encourage better positions.

To get a really concrete sense about how triplet loss solves the problems we saw with hard negatives and hard positives in Triplet loss:
![[Pasted image 20250112172516.png]]
Above: How Contrastive loss "solves" these negative problems
![[Pasted image 20250112172638.png]]
Above: In the easy negative case, we avoid collapsing all of our A's onto a single embedding point. In the hard negative case, we move A2 closer to A1 to make the relative gap between A1 and C bigger.

And now for the positive cases
![[Pasted image 20250112172816.png]]
![[Pasted image 20250112172825.png]]
![[Pasted image 20250112172832.png]]

Let's also talk about some of the complications that Triplet Loss bring to the situation:
- The added computational complexity
	- We have to deal with triplets of examples, as opposed to pairs of examples.
	- The space of possible pairs given all training example is much smaller than the space of possible triplets. It becomes really important doing triplet loss to do intelligent ==[[Triplet Mining]]==: Picking triplets that help your model learn! When the space of triplets is larger, we need to get better and picking good example to help our models learn.
		- ==Avoid redundant triplets== (Doesn't help it learn if the loss is already zero)
		- ==Prioritize hard triplets== (Typically those where for whatever reason the negative is closer to the anchor than the positive is; we'd like to focus very much on these to learn from the most egregious errors!)
		- ==Prioritize semi-hard triplets== (Semi-hard triplets being those in the category where the positive is closer to the anchor than the negative, but not by the margin that we want it to be, so we're still uncomfortable about those.)




---------

### Claude:
==Hard Negatives== are different-class examples that look similar; like two different people who happen to look alike. With basic contrastive loss, we treat all negative pairs the same way, trying to push them apart my *at least* the margin distance. But this doesn't capture an important intuition: We should focus more on distinguishing between faces that actually LOOK similar (hard negatives) than on faces that are obviously different (easy negatives). When we treat all negatives equally, the model might waste effort pushing already-distant pairs further apart instead of focusing on the challenging cases.

==Hard Positives== are the flip side of these. These are same-class examples that look quite different, like photos of the same person from vastly different lighting conditions. Basic contrastive loss tries to push all positive pairs to have zero distance between them, but this can be counterproductive, forcing the model to map very different-looking mages of the same person to exactly the same point in embedding space, which makes the representations less robust and less able to capture natural variations.

This is where ==triplet loss== comes in with an elegant solution: Instead of looking at pairs in isolation, it looks at three examples at once:
- Anchor 
- Positive example (same class as the anchor)
- Negative example (a different class from the anchor)

The key insight is that triplet loss only cares about relative distances -- it says: "The distance between the anchor and the negative should be greater than the distance between the anchor and the positive, plus some margin." Mathematically:

```python
def triplet_loss(anchor, positive, negative, margin=0.2):
	pos_dist = euclidean_distance(anchor, positive)
	neg_dist = euclidean_distance(anchor, negative)

    # We want: neg_dist > pos_dist + margin
    # Or, equivalently: post_dist - neg_dist + margin < 0
    loss = max(0, post_dist - neg_dist + margin)
    return loss
```

This handles both the hard negative and hard positive approaches better:
1. For hard negatives (where neg_dist is small): The loss is larger, making the model focus on putting these challenging cases apart.
2. For hard positives (where pos_dist is large): The model only needs to ensure that this distance is still smaller than the negative distance -- it doesn't force them to be exactly zero.
