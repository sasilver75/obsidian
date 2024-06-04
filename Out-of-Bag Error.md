A method of measuring the prediction error of [[Random Forest]]s, boosted decision trees, and other ML models using [[Bootstrap Aggregation]] (Bagging).

Bagging uses subsampling with replacement to create training samples for the model to learn from.
- ==OOB error is the mean prediction error on each training sample $x_i$  using only the trees that ***did not*** have $x_i$ in their bootstrap sample.==

It's a good test for the performance of the model, and can be seen as an alternate way of doing leave-one-out [[Cross-Validation]], with the advantage of the OOB method being that it requires less computation and allows one to test the model as it's being trained.