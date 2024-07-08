The number of values in a study that are free to vary -- often calculated as the number of observations minus the number of constraints or parameters estimated from the data ("observations minus constrains").
- Intuitive explanation: Each observation provides a piece of information, and each constraint or estimated parameter uses up one piece of information; the remaining "free" pieces of information are the degrees of freedom.
- In a system of equations, degrees of freedom represent the dimension of the solution space.
	- For considering the sample mean of $n$ numbers, once we know n-1 deviations from the mean, the last one is determined (because it must make the sum of deviations zero), so there are $n-1$ degrees of freedom.

In statistical modeling, additional parameters "use up" degrees of freedom from the data; more degrees of freedom *generally* lead to more reliable estimators -- models with too many parameters relative to observations have few degrees of overfitting, since the model has little "free" information to generalize from.

---
- The deep learning paradigm in which we have highly-overparametrized models sort of spits in the face of this; It suggests that our models should be overfit, but they actually generalize well! 
- Training procedures like SGD can act as implicit regularizers, which can reduce the ***effective model complexity*** despite high parameter counts.
- In some views, over-parametrization helps in finding good minima during optimization, and makes the loss landscape easier to navigate.
---
