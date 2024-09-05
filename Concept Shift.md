- This is when your probability of y, given x,  (the relationship between inputs and outputs) is different from train and test time ... but the distribution of inputs themselves do not change.
- $P_{train}(y|x) \neq P_{test}(y|x)$ , but $P_{train}(x) = P_{test}(x)$ 

Suppose we have a two-class classification setting, and we're drawing the two classes across two features.
![[Pasted image 20240629232813.png|300]]
What if at testing time, the *boundary* changes (the points stay in the same place)

![[Pasted image 20240629232838.png|300]]
It's the same distribution of data (with respect to (x1, x2)), but the labels have changed, as we "moved the boundary"!

It's hard to find examples where the input data distribution *does not* change from train to test time, so there's often [[Covariate Shift]] involved as well.

Examples:
- The rating for a song on Spotify
- Predicting the popularity of celebrities based on some features, and a celebrity's popularity goes down or up when they do {some public action} (maybe the input data you're using hasn't changed, but the way people)
- Stock price (If you're predicting a company's stock price based on some fundamentals about the company)

((This in the real world feels like some input variables *have changed*, but they're not input variables that you've captured in your training set.))