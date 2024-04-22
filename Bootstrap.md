Statquest video: [Link](https://www.youtube.com/watch?app=desktop&v=Xz0x-8-cgaQ)
Wikipedia: [Link](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

A powerful and versatile method for assessing the uncertainty of a statistical estimate, involving resampling with replacement from a dataset and recalculating the estimate many times in order to get a distribution of the estimate.

We randomly draw observations from the original dataset with replacement (meaning the same sample can be selected more than once), creating "bootstrap samples" which are the same size as the original dataset, but might contain duplicates.

For each of these bootstrap sample datasets, we calculate the statistic or model of interest (e.g. mean, median, regression coefficients). 

This process is repeated many times (e.g. 1000s), resulting in a distribution of the estimated statistics.

The purpose of the bootstrap is to estimate the variability of a statistic without making strong assumptions about the data distribution -- in predictive modeling, it can be used to assess model stability and performance (by estimating now well the model performs on different samples of the dataset).

It's a flexible and simple tool, but it becomes less reliable when the sample size is very small, since the resampling might not capture an adequate amount of variability in the data.