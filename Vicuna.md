Thursday, April 18 (~1.5 months after [[LLaMA]], ~1 month after [[Alpaca]])
Paper: [Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality](https://lmsys.org/blog/2023-03-30-vicuna/)

A 13B [[Decoder-Only Architecture]] language model trained by ==fine-tuning [[LLaMA]] 13B== ==on user-shared conversations from ShareGPT== (a site where you can share interesting conversations). As far as I can tell, it's basically a larger version of [[Alpaca]] that's trained on selected ShareGPT data rather than directly on GPT-generated outputs.

70k English conversations shared by users and scraped from [[ShareGPT]]
- Pre-processing includes converting HTML to markdown, filtering low-quality samples, and splitting lengthy conversations into smaller segments.
- Compared to single-turn datasets, the ShareGPT conversations ==often consist of *multiple turns*== and are thus more useful for training models to leverage the context of the conversation.
	- ([[Nathan Lambert]] says that the ShareGPT dataset contributions ==may have low *average* quality and narrow distributions==, so be careful!)
- The conversations uploaded to this site may still be owned by the users, so their use is potentially problematic.

Abstract
> We introduce ==Vicuna-13B, an open-source chatbot trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT==. Preliminary evaluation using GPT-4 as a judge shows Vicuna-13B ==achieves more than 90%* quality of OpenAI ChatGPT== and Google Bard while outperforming other models like LLaMA and Stanford Alpaca in more than 90%* of cases. The cost of training Vicuna-13B is around ==$300==. The [code](https://github.com/lm-sys/FastChat) and [weights](https://github.com/lm-sys/FastChat#vicuna-weights), along with an online [demo](https://chat.lmsys.org/), are publicly available for non-commercial use.


![[Pasted image 20240418165458.png]]

