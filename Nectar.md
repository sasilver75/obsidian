November, 2023 -- Berkeley
Blog: [Starling-7B: Increasing LLM Helpfulness and Harmlessness with RLAIF](https://starling.cs.berkeley.edu/)
HuggingFace Dataset Card: [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar)


 An "incredible" [[GPT-4]]-labeled ranking dataset called [[Nectar]] (183k chat prompts) used to train a model called [[Starling]].

Nectar is the first high-quality 7-wise comparison dataset, generated through GPT-4-based ranking.
- Diverse prompts
- High-quality and diverse responses
- Accurate labels

Dataset prompts are an amalgamation of various sources (lmsys-chat-1M, [[ShareGPT]], [[Helpful and Harmless|HH]], [[UltraFeedback]], [[Evol-Instruct]], and [[FLAN]]), and responses are derived from a variety of models (GPT-4, GPT-3.5-turbo, GPT-3.5-turbo-instruct, LLama-2-7B-chat, Mistral-7B-Instruct)

The paper notes the difficulty of overcoming GPT-4's positional bias when ranking responses, and has some techniques.

==This blog seems like a great one to read more thoroughly.==

> "The two most popular open preference datasets are [[UltraFeedback]] and [[Nectar]]" - Nathan Lambert while interviewing Sebastian Raschka, (August 1, 2024)
> > Re: "Is it a problem to use off-policy preference data?", "UltraFeedback and Nectar are both off-policy data from a diversity of models, but the closed labs are all using on-policy preference data (preference data collected over generations from the model that's going to be finetuned on the preference data). So this is a challenge for the open-source community." "This might also be the reason why learning rates in DPO have been tending lower and lower... because our distributions are just in the wrong space, so we're trying to be gentler.... alternatively, it could just be that "it's almost done training"".


Summary
> We introduce ==Starling-7B==, an open large language model (LLM) trained by ==Reinforcement Learning from AI Feedback (RLAIF).== The model harnesses the power of our new ==GPT-4 labeled ranking dataset, Nectar==, and our ==new reward training and policy tuning pipeline==. Starling-7B-alpha scores 8.09 in MT Bench with GPT-4 as a judge, ==outperforming every model to date on MT-Bench except for OpenAI’s GPT-4 and GPT-4 Turbo==. We release the ranking dataset [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar), the reward model [Starling-RM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha) and the language model [Starling-LM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha) on HuggingFace, and an online demo in LMSYS [Chatbot Arena](https://chat.lmsys.org/). Stay tuned for our forthcoming code and paper, which will provide more details on the whole process.
> To facilitate more thorough research into RLHF, a high-quality ranking dataset specifically for chat is essential. We release ==Nectar==, a ==GPT-4 labeled ranking dataset composed of 183K chat prompts==. ==Each prompt includes 7 responses distilled from various models like GPT-4, GPT-3.5-instruct, GPT-3.5-turbo, Mistral-7B-Instruct, Llama2-7B, resulting in a total of 3.8M pairwise comparisons==. Considerable effort was invested in mitigating positional bias when prompting GPT-4 for rankings, the details of which are elaborated in the dataset section below.
> Moreover, there is a ==notable scarcity of open-source reward models==. ==We address this gap by releasing our reward model [Starling-RM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha)==, trained with our K-wise loss on the Nectar dataset.


![[Pasted image 20240420020007.png]]