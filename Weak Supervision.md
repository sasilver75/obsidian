A technique where training data is labeled with noisy, imprecise, or incomplete labels, rather than fully-accurate labels.

This is often used when obtaining fully-labeled data is expensive, time-consuming, or impractical.

The idea is to ==leverage various forms of imperfect information to create a useful training set==. Some sources of weak supervision include:
1. Noisy labels: Labels that are correct *most of the time*, but have some errors.
2. Heuristic Rules: Domain-specific rules or patterns used to generate labels.
3. Distant Supervision: Using some external database to automatically label data based on some form of alignment.
4. Crowdsourcing: Aggregating labels from multiple non-expert annotators, assuming htat majority voting or other techniques can mitigate individual errors.
5. [[Semi-Supervised Learning]]: Using a small amount of labeled data along with a large amount of unlabeled data to improve learning.
6. Transfer learning: Using a model trained on a different, but related task and applying it to the task with limited labels.