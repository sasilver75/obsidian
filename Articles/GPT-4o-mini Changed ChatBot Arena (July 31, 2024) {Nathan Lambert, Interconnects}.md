https://www.interconnects.ai/p/gpt-4o-mini-changed-chatbotarena

-----

[[ChatBot Arena]] casts LM evaluation through the wisdom of the crowd. They don't represent a controlled nor interpretable experiment on LMs. It doesn't measure how the best models are improving in clear ways.

ChatBot Arena in reality shows the strongest correlations with:
1. Certain stylistic outputs from language models
2. Language models that have high rates of complying with user requests

- Meta’s AI is succinct and upbeat (something that [has been discussed many times on the LocalLlama subreddit](https://www.reddit.com/r/singularity/comments/1caj1tb/llama_3_is_now_top5_in_leaderboard_arena/)).    
- OpenAI’s style is the most robotic to me. It answers as an AI and contains a lot of information.
- Claude’s style is [intellectual, bordering on curious, and sometimes quick to refuse](https://www.interconnects.ai/p/switched-to-claude-from-chatgpt).
- Nato hasn't used Gemeni enough to know

![[Pasted image 20240802121510.png|500]]

The [[GPT-4o Mini]] model was marketed as "intelligence too cheap to meter" according to Sam Altman; likely distilled from current or unreleased versions of OpenAI's models.

![[Pasted image 20240802121557.png|400]]
No one would have expected 4o-Mini to rank near the top 3 on a test of "absolute peak ability".
There was a little bit of an uproar in the community about these results.

LMSYS went so far as to share a [thread](https://x.com/lmsysorg/status/1816838034270150984) and a [demo](https://huggingface.co/spaces/lmsys/gpt-4o-mini_battles) to specifically show how GPT-4o-mini performed so well on their arena and it paints a clear picture of what the _average_ ChatBotArena user tests.
- Many of these examples show that it's basically a result of style or refusal.
- The ==overall category of ChatBot Arena should be taken with large error bars!==


Notes re: LLaMA 3.1 scoring
1. Open weight models get to operate without a safety filter added (e.g. [[LLaMA Guard]]), which is a major boost.     
2. Meta ai's concise and slightly different, friendly style will help it. Claude's style (and its penchant for refusing) doesn't appeal to the masses who are voting.

In the most recent LS podcast episode, the lead on the LLaMa alignment team said:
> Now the models are getting so good that it's hard to get to some prompts to break them and to compare models and see their edge cases.



## Partial Solutions and Next STeps

1. ChatBot Aren'as built-in harder catergories (HArd Propmts, Reasoning, Math)
2. Private human evaluation, such as Scale AI's newish [[SEAL Leaderboard]].

Both of these have issues, but are likely better than the default, overall aggregate score on Chatbot Arena.

![[Pasted image 20240802122631.png]]

In the near future, I expect a mix of Hard Prompts, Math, and Code to become the default on ChatBotArena. It’s not an easy transition to make.

![[Pasted image 20240802122652.png]]
Unfortunately, Scale’s leaderboard has a ceiling on trust due to the clear conflict of interest where models they’re selling training data to likely have an advantage by being in-distribution for their human raters.





