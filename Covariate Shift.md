---
aliases:
  - Data Drift
  - Distribution Shift
---

A change in the distribution of input variables between training and test data -- a common issue in production machine learning that can happen either gradually (seasonal trends) or very suddenly (viral news).

![[Pasted image 20240420220825.png]]

What is distribution shift?
- When we train on one distribution, and our model at inference time makes predictions on a different distribution. In short, when your input distribution for training and test don't match, but the relationship between your inputs and outputs don't change.

Examples
- Problems where the distribution of data is time-variant (eg finance)
- A dataset for CV on X-ray images that uses a particular brand of X-ray, but then you deploy it and people around the world use different X-ray machines with their own quirks.
- Self-driving cars, when your training data doesn't have any snowy driving conditions, but you have to deal with them during driving.

((Im my opinion, some of these are just distribution mismatch, as opposed to distribution shift/drift. I usually think of this as your deployed model initially matching the test-time distribution, but over time your model becomes stale as the distribution changes))

- We should be monitoring our data during deployment, and looking for outliers.
- We should also be monitoring our model and the predictions that it's making.