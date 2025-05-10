---
aliases:
  - Contrastive Learning
---
References:
- [RitvikMath: Contrastive Loss](https://youtu.be/dC3_IKaBXTk?si=q-FuVdNIXBVF4bFO)

Contrastive loss is a way to teach models to recognize similar datapoints by pulling together similar items in space and pushing different items further apart.

If we were training a model to recognize different peoples' faces:
1. Model takes two face images and converts each into a numerical representation (embedding vector).
2. If images are of the same person (a ==positive pair==), the loss function encourages the embeddings to be close together by minimizing the distance between them.
3. If the images are of different people (a ==negative pair==), the loss function should push their embeddings apart. It tries to ensure they're separated by at least some minimum distance (==margin==).

Example pseudocode:
```python
def contrastive_loss(embedding1, embedding2, is_same_person, margin=1.0): distance = euclidean_distance(embedding1, embedding2)
	if is_same_person: 
		# Pull similar pairs together 
		loss = distance ** 2 
	else: 
		# Push different pairs apart, but only if closer than the margin 
		loss = max(0, margin - distance) ** 2 
	return loss
```

Problem:
![[Pasted image 20250112164442.png]]
[[Hard Negative]]s: Examples for which it's hard for the model to correctly predict A1 and C as negative examples (which they are, in reality). Let's say person C showed up to work wearing an eyepatch and looked somewhat like person A because they were wearing an eyepatch -- oops! 
Contrastive loss would say that we should move A2 closer to A1. This intuitively seems weird to have all embeddings for a given person's pictures to collapse to the same embedding point.
Right now, no matter what we do, we can't move the embeddings for A1 and C further apart... There's an advantage and incentive for moving A2 closer to A1, though! Contrastive loss will choose to move A2 closer to A1 regardless of the story/context about C.


![[Pasted image 20250112164559.png]]
[[Hard Positive]]: Hard for the model to consider them as positive examples. In this case, we can't move A1 and A2 any closer together :\. Would we want to move C further away? Contrastive loss would say no because of the margin logic... but looking at the richer context of the *triplet* of (A1, A2, C), we would argue that we DO want to move C further away in this case, because doing so will implicitly make it look like A1 and A2 are more similar, since their distances will be relatively smaller when we move C from A1. This is lost in contrastive loss, which refuses to move C further from A1, regardless of the story happening in A1. So we lose something by looking at pairs of examples, so we should look at triplets of examples! This is the guiding insight behind [[Triplet Loss]].

-------
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

-----


