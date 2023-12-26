https://developers.google.com/machine-learning/guides/rules-of-ml


### Terminology:
- `Terminology`: The thing *about which* you want to make a prediction.
- `Label`: An answer for a prediction task either produced by an ML system, or the right answer supplied in training data.
- `Feature`: A property of an instance used in a prediction task. 
- `Feature Column`: A set of related features, such as the set of all possible countries in which users might live.
- `Example`: An instance (with its features) and a label.
- `Model`: A statistical representation of a prediction task.
- `Metric`: A number that you care about. IT may or may not be the thing that you're directly optimizing.
- `Objective`: A metric that your algorithm is trying to optimize.
- `Pipeline`: The infrastructure surrounding a machine learning algorithm! Includes gathering data from the front end, putting it into training data files, training one or more models, and exporting the models to production!
- `Click-through Rate`: The percentage of visitors to a web page who click a link in an ad.

### Overview:
- To make great products:
	- `Do machine learning like the gerat engineer you are, not like the great machine learning expert that you aren't!`
- Most of the resources you will face are *engineering* problems! Even with all the resources of a great ML expert, most of the gains come from *great features*, not *great machine learning algorithms!* 
- So the basic approach is:
	1. Make sure your pipeline is *solid*, end-to-end.
	2. Start with a reasonable objective.
	3. Add common-sense features in a simple way.
	4. Make sure that your pipeline *stays* solid.

This approach will work for a long period of time.
Diverge from this approach *only* when there are no more simple tricks to get you any further! Adding complexity will slow down your releases.

Once you've exhausted some simple tricks, cutting-edge machine learning projects might indeed be in your future. 


# Before Machine Learning

### Rule #1: Don't be afraid to launch a product without machine learning!
- ML is cool, but it requires data. Theoretically, you can take data from a different problem and then tweak the model for a new product, but this will likely underperform basic heuristics! A heuristic can often take you far enough to get started!
	- For instance: 
		- If you're ranking apps in an app marketplace, you could use the install rate or the number of installs as heuristics.
		- If you're detecting spam, filter out publishers that have sent spam before.
		- If you need to rank contacts, rank the most recently used highest (or even rank alphabetically, if you want).

### Rule #2: First, design and implement metrics
- Before formalizing what your ML system will do, track as much as possible in your current system!
- For the following reasons:
	1. It's easier to gain permission from the system's users earlier on.
	2. If you think that something might be a concern in the future, it's better to get historical data *now*.
	3. If you design your system with metric instrumentation in mind, things will go better for you in the future! You really don't want to find yourself grepping for strings in logs to instrument your metrics.
	4. You will notice what things change and what stays the same. For instance, suppose you want to directly optimize one-day active users. However, during your early manipulations of the system, you may notice that dramatic alterations of the user experience don't noticeably change this metric.

Note that an ==experiment framework== that lets you group users into buckets and aggregate statistics by experiment is important!
- By being more liberal about gathering metrics, you can get a broader picture of your system. 
- Notice a problem? Add a metric to track it! 
- Excited about some quantitative change on the last release? Add a metric to track it!

### Rule 3: Choose Machine Learning over a *complex* heuristic
- A simple heuristic can get your product out the door. A *complex* heuristic is unmaintainable.
- Once you have data and a basic idea of what you want to accomplish, move on to machine learning.


# ML Phase I: Your First Pipeline
- Focus on your system infrastructure for your first pipeline. It's hard to figure out what's happening with the imaginative machine learning you're going to do if you don't first *trust your pipeline!*
### Rule #4: Keep the first model simple, and get the infrastructure right
- The first model provides the biggest boost to your product, so it doesn't need to be fancy.
- You'll run into many more infrastructure issues than you'd expect:
	1. How to get examples to your learning algorithms
	2. A first cut as to what "good" and "bad" mean to your system.
	3. How to integrate your model into your application. 
		- You can either apply the model *live*, or you can precompute the model on some examples *offline*, and store the results in a table.
		- For example, you might want to pre-classify web pages and store results in a table, whereas you might want to classify chat messages live.
- Choosing simple features makes it easier to ensure that:
	1. The features reach your learning algorithm correctly.
	2. The model learns reasonable weights.
	3. The features reach your model in the server correctly.
- Once you have a system that does three things reliably, you've done most of the work!
- Your simple model provides you with baseline metrics and a baseline behavior that you can use to test more complex models. 
	- ==Some teams actually aim for a "neutral" first launch==: a first launch that explicitly deprioritizes machine learning *gains*, in order to avoid getting distracted.

### Rule #5 : Test the infrastructure *independently* from the machine learning!
- Make sure that the infrastructure is testable, and that the learning parts of the system are encapsulated so that you can test everything around it.
- Specifically:
	1. Test getting data into the algorithm. Check that feature columns are populated, and manually inspect the input to your training algorithm.
	2. Test getting models out of the training algorithm. Make sure that the model in your training environment gives the same score as the model in your serving environment.

### Rule #6: Be careful about dropped data when copying pipelines
- Often, we create a pipeline by copying an existing pipeline, and the old pipeline drops data that we actually *need* for our new pipeline!
- For example, the pipeline for Google Plus *What's Hot* drops older posts (because it's trying to rank fresh posts). If this pipeline were copied to be used for Google Plus *Stream*, where older posts are still meaningful, then that would be a problem!

### Rule #7: Turn heuristics into features, or handle them externally
- Usually, the problem that your ML is trying to solve isn't completely new. There's an existing system for ranking, or classifying, or whatever problem it is that you're trying to solve.
- This means that there's a bunch of rules and heuristics.
- These same heuristics can give you a lift when tweaked with machine learning! Your heuristics should be *mined* for whatever information they have! Those rules create a lot of intuition about the system that you don't want to throw away.
- Four ways to use existing heuristics: ==PICK UP HERE, SAM! I WAS TOO EEPY TO CONTINUE :)==
	1. `Preprocess using the heuristic`. 
		- If the feature is incredibly awesome, then this is an option. 
		- For example, if, in a spam filter, the sender has already been blacklisted, don't try to relearn what "blacklisted" means; block the message! This makes the most sense in binary classification tasks.
	2. `Create a feature`.
		 
	3.  `Mine the raw inputs of the heuristic`.
		- 
	4. . `Modify the label`.
		- 



































