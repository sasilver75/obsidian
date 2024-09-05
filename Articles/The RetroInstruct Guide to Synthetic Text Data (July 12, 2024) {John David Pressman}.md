https://minihf.com/posts/2024-07-13-the-retroinstruct-guide-to-synthetic-text-data/

Author [JD Pressman](https://x.com/jd_pressman) makes some good tweets, fwiw. Good follow.
Here's an [interview](https://soundcloud.com/user-557955426/episode-22-eigenrobot-vs-extropy) of him on Eigenrobot's podcast

----

This post is based on JDP's experience working on the [RetroInstruct](https://github.com/JD-P/RetroInstruct) ("RetroInstruct: Royalty-Free Instruction Data through Backtranslation and Synthetic Methods") dataset.

There's a lot of hype and criticism of synthetic data:
- -: Will synthetic data only be a path to deteriorating models?
- +: Does LLMs improving their math skills with proof solvers mean we're approach hard takeoff?
The reality is more interesting than either!

## Conceptual Background

### The four kinds of synthetic data
1. ==Formal Verification Methods==
	- These methods typically involve a proof assistant or program test suite and are generally used for either making math proof datasets or writing code.
	- Notably, you have either total nor near-total certainty in the correctness of solutions found with these methods.
2. ==[[Back-Translation]] Methods==
	- These methods start with the right answers, and then generate the questions.
	- Diffusion models are base on this principle, where we add successive layers of noise to an existing sample and then teach a neural network to undo the noise.
	- If language models couldn't already write summaries and we had to teach them from scratch, we could use an existing formal summarization algorithm to teach the LM to emulate it by running the algorithm over text to get text:summary pairs, and then perform backtranslation by reversing the pairs so that they imply writing the text by starting from the summary or outline.
	- We could take a source like Wikipedia, break it into chunks, and ask a language model "what questions could this chunk answer," then perhaps even write the chunks in different styles to get a comprehensive Q&A dataset over a much wider range of subjects than any set of human contractors could write!
		- ((This reminds me of things like [[Genstruct]] and [[Web Rephrase Augmented Pre-training|WRAP]]))
3. ==[[Rejection Sampling]] and Classifier Methods==
	- These use a combination of a generator/prompt that's usually write, and a classifier which sorts out the failures in the batch.
	- With a good model and classifier, you can get very good results. If you have a generator that gives correct samples 80% of the time and a classifier which is also incorrect 80% of the time, your overall accuracy could become as high as 94.12% if you only accept sample the (bad) classifier agrees with.
4. ==Prompting Methods==
	- They're functionally rejection-sampling methods but without rejection and without the classifier.
	- If the usual prompt a user types in for some tasks gets it right 60% of the time, but clever prompting or automatic search with a framework like DSPy lets us find a context in which it becomes 90% right, we can take the results from that and put a normal looking prompt in front of it to make the usual performance of the model closer to 90%.
		- ((Is there a name for this technique? Reminds me vaguely of the "ghost attention" technique from the LLaMA 2 paper, in the sense that you're doing a little bit of a "bait and switch"))

### The mental motions of synthetic data design
- Useful synthetic data pipelines are going to incorporate elements of *all* of the above to get good results! Designing the pipelines is almost like a game with rules, and a series of steps can form a valid or invalid pipeline. Let's go through a few examples of pipelines:

#### Code Debugging Dataset
1. Formal Verification: Take a large corpus of existing permissively-licensed code and filter it down to samples which are relatively standalone -- ones that do operations on raw strings, numbers, etc. We want to eliminate code that relies on lots of libraries with unpredictable behaviors or that rely the behavior of a stateful system outside the context window.
2. Prompting: The functions we have can be debugged from short context, with the content of the function alone. We can prompt an LLM for test case input:output  pairs to validate our corruption pass against. The idea is to perform backtranslation by introducing *deliberate errors* into known working code, and then teach the model to go from errors to the original known-correct code.
	- To improve the odds that our LLM is in fact breaking the code when we prompt it to, we can start by making a small test suite for each piece of code that we're trying to break.
	- So we tell the LLM to write a series of inputs and expected outputs for each function, and run the test suite against that function, insisting on full code coverage... but this isn't going to guarantee our test suite is correct, so this isn't quite formal verification.
3. Prompting: Now that we know that our functions "work" because of our test suites for each function, we ask our model to break each of the functions with a type of bug. Perhaps we tell it the kind of bugs we want to introduce (eg syntax error, type error, logic error) by randomly selecting from a list, or maybe we let it pick the bug it wants to introduce to the code. We want to break them in a way that verifiably makes the test suite no longer pass. We might include the test in the prompt with the explicit instruction to introduce a certain type of error such that the test suite no longer passes. We only retain samples where the test suite is in fact broken.
2. Backtranslation: Now we have a corpus of simple low-state logic-heavy functions, and we have test suites of some unknown quality for those functions, and broken versions of the functions that don't pass the test suites. 
	- To perform backtranslation and create the debugging dataset, we simply reverse causality! We start with the broken functions (and maybe an error message from the test suite), and have the chat assistant learn to reply to the  information with the "corrected" (known to be correct to begin with) code that resolves the problem.
3. Prompting: If we want to use this dataset in an instruction-tuning context, we still need to give it the introduction to the problem telling the model what it is we want to do. This is important so that at inference time the model knows that it should reply to similar requests with the appropriate solved program. Author does it like Flanv2 does it, where you write several variations on the question or prompt format and if relevant insert necessary information into the question variant using template variables. Or you could have an existing LLM write some variants of the question. The purpose is that the model generalizes to different versions of the same question ((Sort of a type of [[Consistency Regularization|Consistency Training]])).

At the end we should have
```
{INSTRUCTION_TEXT}

<code>
{CORRUPTED_CODE}
</code>

<errorlog>
{TEST_SUITE_FAILURE}
</errorlog><|end|>
{ORIGINAL_CODE}
```
```


```

We can imagine variations of this template, like one where the test suite failures are generated by the model rather than given by the user:
```
{INSTRUCTION_TEXT}

<code>
{CORRUPTED_CODE}
</code><|end|>

<errorlog>
{TEST_SUITE_FAILURE}
</errorlog><|end|>

{ORIGINAL_CODE}
```


#### Multi-turn Instruction Data
- Author is often told that there's a critical shortage of good open multi-turn instruction data!
	- He thinks it's weird, because creating good synthetic multi-turn data isn't that hard in principle.
- We should ask: "What are reliable contexts in which I can demonstrate 3 or 4 turns of conversation?"
	- This could be us stretching our bug-introducing pipeline above into multiple stages, so that the LLM learns to break a particular test on each pass, and then make multi-turn data where the LLM fixes each individual bug one at a time.
		- Downside is this teachers the model to be "lazy" and refuse to fix more than one bug at a time...
	- Could use an embedding model for code to find semantically similar function, and then have a multiturn set where a user "asks" the model to "transform" them into an arbitrarily chosen final function over multiple interaction to simulate "getting closer to what the user wants."
- Here's a fuzzy strategy for poetry based on backtranslation (where a user supplied a bad poem, asks for the genre to be changed, and then asks for the poem to be improved)
	- Prompting: The basic structure is going to have a rhythm like bad -> genre change -> original. So first we want to find a prompt that changes the poem's genre. We don't need a prompt to preserve rhyme structure because the next step after this is supposed to be bad poetry in the first place!
	- (optional Rejection Sampling): To make our genre changes bedder, we could use a prosody analysis algorithm or speech synthesis model to measure the awkwardness of the prose when spoken. Few shot logit evaluator prompts could also be used to get a better sense of whether a poem preserves rhyme structure of the original or not.
	- Prompting: Once we have the poems with changed genres, we want to further prompt the model to make them bad. 
	- Prompting: We'll also need some glue text like the user's instructions/questions that frame the prompt
		- "I'm having trouble writing this poem, could you  help me?"
		- "I think the real problem is that this poem is about love, can you make it about friendship instead?"
			- (You should retain these specific genres you had the model change the original poem to so you can put those genres into the generated user instructions you wrap the poems with)
		- "Of course I can help you with your poem ..."
	- Backtranslation: Once you have all 3 turns, you can perform backtranslation by putting it into a template in reverse order to how you made it!
```
User

{GENRE_CHANGE_INSTRUCTION}

{BAD_POEM}

Assistant

{ASSISTANT_PLEASANTRY}

{GENRE_CHANGED_POEM}

User

{POEM_IMPROVEMENT_INSTRUCTION}

Assistant

{ASSISTANT_PLEASANTRY_2}

{ORIGINAL_POEM}

```
I think the key is to think of way of instruction generation that don't teach the model bad behavior.

## Easy Prose Repair Diffs
- (Skipping this section, lots of details)

## Concluding note on Frameworks and Automation
-  Author has focused more on developing a technique than a framework, because his honest impression is that synthetic data is something like a form of software development that won't be fully automated until we automate software authorship in general!
- In the short term, he can imagine automation of particular design patterns and workflows, but you can't really automate a process until you're familiar with it. Synthetic data is still in the exploration phase.
	- You can assist with the process of learning these patterns that can be formalized by designing, implementing, and publishing your own synthetic data pipelines. 
	- If even just a few hundred people did this and their work was put together it would well outstrip the task diversity in FLAN.

None of the currently well advertised ways to contribute to deep learning are particularly accessible:
- Model tuning is a malthusian winner-take-all rat race where people scavenge the same sources over and over for marginal improvements to model performance. It's notable that Hermes Nous, the overnight champion got there by making synthetic datasets with instruction models instead of picking over the same stale data every other public model team was using.
- Most new techniques require the author to be well versed in a corpus of dense math papers full of inscrutable symbols from the perspective of an undergraduate. It's simply not attainable for people with a normal software development background unless they're willing to put in a lot of study hours and ingratiate themselves to people who already know.
- Benchmarks are in theory open to anyone with a clever idea for measuring model performance but in practice are massive undertakings with an ideal set size of thousands of rows. They are not a hobby software project, but more like a full time job.
- Projects like OpenAssistant require a big name to back them for coordination purposes and don't seem to have a lot of impact relative to the effort that went into them. Teknium, founder of Nous Research and current model tuning champion openly mocks how bad OpenAssistant is on his Twitter account which has to discourage contributions.
As far as I know synthetic data is the only truly accessible way to contribute to open source AI for someone with a normal software development background. Since the bottleneck to automating most pipelines is subjective evaluation of whether a particular prompt works or not I expect it to remain an available avenue to contribute for at least a while.

