A tool for [[Outlier Detection]]

A neat algorithm is something called an [[Isolation Forest]] (cool name)
- If you have a random decision tree, and if take all of our data and see how far down the decision tree we have to go to end up with only that single piece of data, isolated from the rest of the data... the more a piece of data is an outlier, the shallower you'll have to go down the decision tree!

![[Pasted image 20240629225522.png|200]]
Given a dataset, choose a feature at random, choose a cutoff at random, and then create that decision boundary in your tree.
![[Pasted image 20240629225627.png|300]]
Making up some draws here, but see that we're already isolated that point on the right that isn't part of the two clusters? Nice!

Think: Would it make sense to apply this to image data, where we're working with raw pixel values (lots of features, where every feature is a pixel value for some channel)?
- No! You probably want to embed your image into a lower-dimensional space where the place the image ends up in the embedding space means that similar images end up near eachother.