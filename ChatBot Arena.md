June 9, 2023 -- UC Berkeley, UCSD, Carnegie Melon, Stanford
Paper: [Judging LLM-as-aJudge with MT-Bench and Chatbot Arena](https://arxiv.org/pdf/2306.05685.pdf)
Leaderboard: [Link](https://chat.lmsys.org/?leaderboard)

This paper actually introduces two banger evaluations: [[MT-Bench]] and [[ChatBot Arena]]!

The latter is described as a "Chatbot Battle Platform!" -- it's a crowdsourced platform developed by [[LMSYS]], featuring anonymous battles between chatbots in real-world scenarios; Users engage in conversations with two chatbots at the same time, and rate their responses based on personal preferences -- we create a leaderboard out of this too!

Arena-like systems that depend on the wisdom of the crowd works pretty well for quantifiable things where you have a clear metric (eg estimating the weight of a pig, number of marbles in a jar)... but it's hard to get reproducible results from it. While you avoid some of the biases of [[LLM-as-a-Judge]], you don't avoid all bias. Raters often prefer models that seem to *like us* and *agree with us!* Raters prefer models that are super assertive/confident about their answer, even if that means they're wrong (eg a cautious true answer is sometimes dispreferred to a confident wrong answer). Most raters demographically are men from the United States, which isn't representative enough. Most interactions before the user gives a rating turn out to be single-turn interactions, so multiturn performance isn't meaningfully evaluated (though it often isn't evaluated in automatic evaluations either).

-----


Abstract
> Evaluating large language model (LLM) based chat assistants is challenging due to their broad capabilities and the inadequacy of existing benchmarks in measuring human preferences. To address this, we explore using strong LLMs as judges to evaluate these models on more open-ended questions. We examine the usage and limitations of LLM-as-a-judge, including position, verbosity, and self-enhancement biases, as well as limited reasoning ability, and propose solutions to mitigate some of them. We then verify the agreement between LLM judges and human preferences by introducing two benchmarks: ==MT-bench==, a multi-turn question set; and ==Chatbot Arena==, a ==crowdsourced battle platform==. Our results reveal that strong LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving over 80% agreement, the same level of agreement between humans. Hence, LLM-as-a-judge is a scalable and explainable way to approximate human preferences, which are otherwise very expensive to obtain. Additionally, we show our benchmark and traditional benchmarks complement each other by evaluating several variants of LLaMA and Vicuna. The MT-bench questions, 3K expert votes, and 30K conversations with human preferences are publicly available at [this https URL](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).


# Paper Figures


# Other Figures

![[Pasted image 20240430112429.png]]
The idea was: What's in this sector of the evaluation graph? Can we build something for it?
Issues with Ground-truth benchmarks: Can't handle open-ended tasks (eg writing style), and can't capture nuances in human preferences (eg brevity)