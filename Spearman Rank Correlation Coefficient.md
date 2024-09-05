---
aliases:
  - Spearman's Rho
---


Compare: [[Pearson Correlation Coefficient]] ("Correlation")

Also known as $\rho$ , it is a non-parametric measure of rank correlation.
- Here, non-parametric measure means that it doesn't make any assumptions about the distribution of the data, whereas parametric methods assume that data follows a certain distribution.

It assesses how well the relationship between two variables can be described using a **monotonic function**.

Unlike the [[Pearson Correlation Coefficient]], which measures linear relationships, Spearman's rho evaluates the strength and direction of the association between two ranked variables.

Key points:
1. **Rank-Based:** It works with the ranks of the data rather than the raw data values. This makes it suitable for ordinal data or for data that do not meet the assumptions of Pearson correlation (e.g., non-linear relationships or non-normally distributed data). Here, "rank" of a variable refers to the position of each value within its dataset when sorted in asc/desc order.
    
2. **Monotonic Relationship:** Spearman's rho measures how well the relationship between the variables can be described by a monotonic function, which is a function that either never decreases or never increases. It does not require the relationship to be linear.
    
3. **Range:** The Spearman correlation coefficient ranges from -1 to +1.
    
    - A coefficient of +1 indicates a perfect positive monotonic relationship.
    - A coefficient of -1 indicates a perfect negative monotonic relationship.
    - A coefficient of 0 indicates no monotonic relationship.