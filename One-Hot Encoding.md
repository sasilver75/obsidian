A type of (extremely) sparse vector in which we represent a categorical variable as a vector of all zeroes, with a single 1.

For example, we might represent a *Gender* variable as $[0,1]$ 

This also helps avoid the problem of ordinality, which can occur when categorical variables have a natural encoding (e.g. representing "small," "medium", and "large" as 1, 2, 3, rather than as a one-hot-encoded vector).