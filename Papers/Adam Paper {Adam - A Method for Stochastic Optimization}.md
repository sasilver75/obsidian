December 22, 2014
Link: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
#zotero 

"Adam" -> "Adaptive Moment Estimation"
Meant to combine the advantages of both [[Adagrad]] (Duchi, 2011) and [[RMSProp]] (Hinton, 2012), and be a "simple and computationally efficient algorithm for gradient-based optimization of stochastic objective functions."

It's just an optimizer that incorporates things that *I'm* going to term as velocity, acceleration, and friction. It's adaptive momentum. Exactly how it determines this isn't super interesting to me at the moment.

I believe this might actually be the most cited paper of all time -- or at least of the decade, having been signed (as of April 2024) 175k+ times.

Abstract
> ==We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments==. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. ==The hyper-parameters have intuitive interpretations and typically require little tuning==. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that ==Adam works well in practice and compares favorably to other stochastic optimization methods==. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm.
