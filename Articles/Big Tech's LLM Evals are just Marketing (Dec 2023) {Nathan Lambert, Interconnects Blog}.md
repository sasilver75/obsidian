#article 
Link: https://www.interconnects.ai/p/evals-are-marketing

------

December 13, 2023

Google announced the Gemeni model suite last week, and OpenAI/MSFT said something in response to the effect of: "GPT-4 is better than Gemeni Ultra because it scores .06 higher on one esoteric benchmark!"


From Microsoft papers:
- *Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine*
- *The Power of Prompting* (Nov 2023)

![[Pasted image 20240326200701.png]]
Above:
- We can see incremental improvements with GPT-4's MMLU score using more and more sophisticated prompting techniques, and that at the "best prompting technique (MedPrompt, here)" the GPT-4 model barely edges out Gemeni Ultra using 32-shot [[Chain of Thought]]

Of course, ==Microsoft/OpenAI can't reliably compare these two models, each using different prompting techniques -- they don't even *know* the specifics of how Gemeni Ultra was prompted, just that it was 32-shot CoT!==


And Google is guilty of this too! From the Gemeni launch:
![[Pasted image 20240326201038.png]]
Again, ==Google is saying that, with their specific prompting technique (32-shot CoT), it's better than "Human Experts" (?) and 5-shot GPT-4! That's clearly an annoying comparison== -- they don't have the ability to really evaluate/compare with their competitor!
- And annoyingly, Company A showing their model "*crushes*" Company B might have a positive effect on Company A's stock, so the incentives are quite misalligned!

Without access to the training data or code, little can be said with confidence as to why LLM eval scores are different.

If you want to be informed about which LLM is the one for you, sit down and chat with some LLMs, and you'll know the answer pretty quick.

==Prompting and evaluation don't mix well!==
- Prompting/[[In-Context Learning]] is most useful for tailoring the model's performances to specific needs and styles; it's a fantastic tool for model users. But there's also an art to it, and different models may (??) respond to different prompting styles.
	- There are manual and automatic prompting techniques
- ICL is the fastest and cheapest way to personalized models.



























