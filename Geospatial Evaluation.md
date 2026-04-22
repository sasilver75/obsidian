References:
- Video: [Adam Stewart's AI4GEO 2025: Lecture 3: Evaluation](https://www.youtube.com/watch?v=ShVZ5COTNo4)


Many situations in which we would like to *evaluate* a model:
- Given a model, how likely will this model be to make a correct prediction?
- Given ten models, which model is best, and why?
- When does the model make mistakes, and why?
- Will the model work on different kinds of data?

Qualitative analysis can be fast and useful, but can very, very subjective.
Quantitative analysis (done correctly) is more objective.

In practice, you typically want to use both: Sometimes you'll have two models that are quantitatively similar, but when you actually use them, you notice qualitative distinctions between the models.


![[Pasted image 20260422140410.png]]

![[Pasted image 20260422140638.png]]

![[Pasted image 20260422140800.png]]
[[Stratified Sampling]]: Guarantees that it's basically the same distribution every time. Model won't say: "I've only seen Urban areas, so I'll always predict Urban." Makes sure that you have similar distributions.
- Can optionally shuffle within each class.


![[Pasted image 20260422140943.png]]
How well does the model perform on an entire region that it hasn't seen before, e.g. splitting on geographic groups of data?
- A country being either in training or testing but not both at the same time
	- Training on every continent but NA, and then testing it on NA. A good way to see which continent is the most different.
- Can combine this with stratified and grouped.
- For instance, you might want to do a plate tectonics split!
	- ![[Pasted image 20260422141300.png]]
	- For some reason, Adam didn't want one plate to be in training and testing, so he did a plate tectonics split. It got rejected and they didn't like it, because the pacific plate was huge, and it just so happened that predicted that the entire pacific plate is just one big anomaly...  What he should really do instead is more of a grid-based approach, where he's not completely avoiding the plate... (? Something specific).
		- Pacific plate is way bigger than all the others; Some other plates look smaller, an artifact of the mercator projection.
		- The big pixels near the antarctic are actually not very big, we shouldn't count those as much as the ones near the equator.
	- ![[Pasted image 20260422141447.png]]
	- A more commonly-used grid approach; Chop it into a grid and randomly assign grid cells to different training/validation splits.

Time Series:
- Most metrics make IID assumption in data (random split), but this rarely holds in the real world.
	- Temporal datasets are an example of this. Predicting values 3 years from now is very hard.
- How to measure out-of-distribution performance?
- Use a sliding window approach!
- ![[Pasted image 20260422141606.png]]

![[Pasted image 20260422141918.png]]

![[Pasted image 20260422141931.png]]
[[Mean Squared Error]] [[Root Mean Squared Error]]

![[Pasted image 20260422142027.png]]
[[Pearson Correlation Coefficient]]

![[Pasted image 20260422142232.png]]
[[Coefficient of Determination]], R-Squared.
- Don't write it as r^2, that's different!


![[Pasted image 20260422142358.png]]

![[Pasted image 20260422142539.png]]
[[Confusion Matrix]]

![[Pasted image 20260422144513.png]]
[[Precision]], [[Recall]], [[Precision-Recall Curve]] , [[ROC Curve]], [[ROC-AUC]]


![[Pasted image 20260422144624.png]]
[[F1 Score]], etc.
- High when both P and R are high
- Low when either P or R are low
- It's very easy to get a perfect score on either P or R by always guessing the same thing. It's much harder to get both to be good.

![[Pasted image 20260422144716.png]]
[[Jaccard Index]], can also be applied to image segmentation (though P, R, F1 score is also used there, and is preferred by speaker)

![[Pasted image 20260422144755.png]]
Overall accuracy is probably 95%, but for certain people it's really bad.
- The model isn't racist, it just didn't see a balanced training distribution.
- A lot of the older models, they'd just take pictures of the people making the model, and use that. And there's a lot of white male engineers.

![[Pasted image 20260422144912.png]]
Can also do weighted metrics
- If I'm predicted cropland, some types of crops are perhaps more important than others. Things like lentils are relatively less common and have less impact... so maybe you want to weight things to match that!


![[Pasted image 20260422144955.png]]
There are many more!








