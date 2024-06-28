https://www.youtube.com/watch?v=R9CHc1acGtk&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5&index=5

---

![[Pasted image 20240627190334.png]]
Remember this count matrix that we saw yesterday; it's an un-normalized joint distribution, basically.
- The 32 means that there were 32 examples that were *labeled* as cow that were *actually* dogs.

![[Pasted image 20240627190517.png]]
Say we had this count matrix, where all of the off-diagonals (mislabels).
We take the 50+50=100 things, and we can get them ...
We can remove them from the dataset and get a better model
What about another way, that has a lot to do with dataset curation?

Unlike in school where you have a dataset where you have ot predict 3 or 4...
You can create a new class called 3-4! So now you just have three classes (1,2,(3,4)), or something.
Or have a second model that then tries to discriminate between the two, once you've predicted an image to be a (3,4)

So this lecture we're thinking about how we change our dataset

Other things we could do:
- If the upper-right 50 were instead a 20...
	- Now we have a different rate of error in each direction. 
	- This chart is more informative that
	- this might happen when we have a "is a" relationship; a missile is a projectile. Someone frmo california is an american. It's more likely that someone in california is mislabeled as being from the us, versus someone in the us being labeled as being in california, because there are 50 states

Other one
![[Pasted image 20240627194350.png]]
Say these were the frequencies of our classes
- We have 4 classes with a lot of data, and many classes (7) that have just a little bit of data
We can train a model to do well on all of them, or you could (from the dataset perspective) join all of these classes together, and make it one clutter class.
- You could then train another model that, once something is predicted as clutter, we give it to this second model to predict the resulting class.


![[Pasted image 20240627194556.png]]
Ranking the off-diagonals in terms of what occurs most
- Mislabeling a missile as a projectile is the most mistaken
	- Note that projectle is also mislabaeled as a missile often too
- Notice that this is an "is a" relationship
- This is a broken dataset! there's no single label that's necessarily "more true"
- ==Interstingly, imagnet actually has two classes in it that are teh same class -- that's just a mistake! There are two "maillot" classes that are different classes!==

So
- Discovering issues in the dataset (classes labeled as other classes)
- Maybe we'd like to merge these classes; it's supposed to be a single-label classification dataset, but there are records with multiple true labels.

![[Pasted image 20240627194852.png]]
This is from Imagenet too.
The one in the left has a curved tail! It's up to the datset curator to choose whether your model needs to be able to discern so finely.
We can use our appropach to identify these two related classes (liekly with mislabels) and perhaps merge those classes

![[Pasted image 20240627195115.png]]

![[Pasted image 20240627195223.png]]

![[Pasted image 20240627195325.png]]
There aren't many cows at the beach in our training dataset, is why!
The problem of "Spurious Correlation"! Our models are looking for *any sort of purchase* to help them predict the label of a tensor of numbers (that we call an image).
- A Spurious Correlation is one that's present in the data that you're training your model on that doesn't remain in the data that you're going to deploy your model on, like in the real world.
- ML models are almost cheaters looking for shortcuts; they're trying to find any sort of pattern in the training data that's highly correlated with the labels. It has NO KNOWLEDGE for the real world, and so will latch onto anything that's available -- you need to be careful about what kind of patterns might be present in a training dataset.

Spurious Correlation are an instance of Selection Bias
- Where the training data isn't fully representative of the real world, or of deployment distributions. It's a distribution mismatch between the training dataset and the the real-world deployment distribution.

Added some more information to the new [[Selection Bias]] note.


---

If you goal is 95% accuracy, how much data do we want? Let's assume that we've already got some data of sample size N with some performance.
- Can we estimate how much data we need?

Student: "We can use theoretical generalization bounds?"
- Many of the generalization bounds will be pretty vacuous for modern ML




