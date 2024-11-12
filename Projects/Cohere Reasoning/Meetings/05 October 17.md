

It seems for the 250 (500?) examples that I did, we didn't have any successful solutions.
- Could my perturbations be too difficult? IE the weak-completion model was just *too dumb?*
- I noted that it was pretty rare for the strong-completion model to be able to arrive at the correct answer anyways. Would we assume that models have a higher propensity to recover when tackling questions that it can easily handle? Should we consider an easier dataset, like [[GSM8K]]?
- Do I just need to scale my N higher?
- Let's look at some examples in my little UI. It's easy to spin it up on your computer.

Price is pretty expensive for what I'm doing...
- Iterations of weak solution and verification.
- Straight-shot completion and verification.
- Strong completion of weak prefix.
Burned like $20 yesterday, it seems like. Should I worry about that?

For other inference providers, it seems like Together has a chat completions feature similar to the one that Anthropic offered. 
- Asking in their discord if it's doing under the hood what I'd hope for it to do -- or I can just test it out.
- I don't mind covering some small charges, but ... funding for that?

----

Maybe the 250 are just too hard, if the strong completer can't straight shot the completions.

For a large subset of questions, which percentage can it get right in a straight shot? Ideally use a few samples per question. Because I imagine it will be somewhat bimodally distributed.

Otherwise, we might want to use a weaker dataset.
uest
The absolute strength of the model/dataset doesn't matter; calibration of the model (so it's just hard enough to be able to do some well, some not well, some in the middle.)
- Ultimately it's a 
- Whatever prompts we have sourced from every dataset, for each model, we just need to check their natural straight shot completion rate on the datsaet
	- Do we do normalization? That seems sort of sketch and unfounded though :D 
	- What could be done is that we at least report it, which help interpret results
		- So it turns out that (eg)
			- For o1, it solves a bunch of stuff straight-shot.... and maybe it's also good at recovering from perturbations. This might just mean that when you put this in a figure, you can just say that this had a higher recovery rate, but this is an intrinsically (relatively) easier dataset for this model.


Our experiments with Cohere models will be well-controlled, because we know everything we need ot know about he models. as well as the API layer above models.
But the reality is that maybe some model provider is doing something weird behind the API, behind OpenRouter.

I know for sure that NuminaMath has some trivial ones too!

The binning... at first, will need to be kind of fuzzy (here are some eassier, harder probleems)
The first thing we want to do is replicate the experiment, gut throw in some easy problems too.


1. Get an evaluation for the strong completer model of distribution of correct straight-shot solutions for some medium N of problems.
2. Fuzzy binning of problem by difficulty.
3. Once we have some easier problems, rerun the experiment.
	1. Hopefully the expected result is that now there are some that can self-correct.
	2. Figuring out that pattern is the thing...
	3. If it still can't self-correct...
		1. The way we're perturbing might be too strong


Todo:
- This graph regenerated that we have
- Another graph overlaid with teh intrinsic strong performance
- Do we want the weak completer straight shot performance as well
- You can do the on-policy perturbation tooo: Let's say we have a roughly 50/50 correct/incorrect on the storng completer.... We have a weak completer prefix as a perturbation, check how the strong does. But if the strong completer is shaky on some of these, then that means that you can use the wrong... Strong completer prerformance, we're taking multiple completions per promtpt (eg 3-5 ompletion attenmpts). Basically for anything where it's in the middle... where 2-5 are corect or so... Take one of the wrong traces and truncate it to make it a perturbation prefix.


For a prompt where the strong completer naturally is kidn of 50/50, you can just sample when it's wrong, truncate it... Which is the way to have
While we're doing this.... it might look very similar to try the on-policy stuff at the same time, if serindipidously we happen to come across datapoints wher the strong completer is naturally sometimes right and sometimes wrong.


The key piece of info: Given a dataset where we know the strong completer intrinsic performance on it, and wed know it has a reasonable distribution


For perturbation: Cap your number of samples as something quite reasonable (10). If we have problems tha are sufficiently easy that we can't perturb them, leave them aside. 
- Could use Jimin's strategy to perturb...
- If the problems ar so easy that the weak perturber can always solve them, they might just be not interesting to study.
