Also called Confounding, or Distribution Shift

Your model may fit to *spurious correlations* in the training dataset, if the distribution that you train on is meaningfully different from the distribution that your model encounters in the real world.

It's really hard to deal with, and something that you should always be thinking about.

You should select your validation data to be *most representative* of the data that you'll see during deployment (rather than a random training/validation split, think carefully about how your data might look different in the future, and try to select data that represents that, and remove that from the training data so that you see how your model will suffer under that selection bias).

![[Pasted image 20240627210654.png]]

Examples
- Time-based machine learning where you collect data in the past, and you have to do inference on "future" data when deployed; there might be patterns in the past that are no longer present in the future, or vice versa (a big problem in stock trading, for instance).
	- Use your most recent data for your validation data.
- If you over-curate or over-filter your dataset because you don't want to deal with (eg) messy data, but you actually *will see* messy data during deployment.
- Rare events that might not make it into your training data. For self-driving cars; it's difficult to collect data about what will happen during a car crash. 
	- Intentionally over-sample rare events for your validation dataset (making them even rarer in your training dataset).
- Convenience-based collection of training data (eg you're a grad student doing some king of survey, and you only survey your friends because it's easy).
- Location-based selection bias; you're going medical ML, and training data only comes from 3 hospitals, but you want to deploy a model to the entire country.
	- Hold out all of the data from some locations that you mean to use as a test.

