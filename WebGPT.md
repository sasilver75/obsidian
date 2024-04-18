December 16, 2021 (GPT-3 was released in June 11, 2020; ChatGPT in November 30, 2022) -- [[OpenAI]]
Blog: [WebGPT: Improving the factual accuracy of language models through web browsing](https://www.amazon.com/gp/video/storefront/ref=atv_hm_hom_legacy_redirect?contentId=IncludedwithPrime&contentType=merch&merchId=IncludedwithPrime)

An OpenAI fine-tune of [[GPT-3]] that's trained to do internet searches.
A much earlier example of language model tool-use than either [[Toolformer]] or [[Gorilla]].

Abstract
>We’ve ==fine-tuned GPT-3 to more accurately answer open-ended questions using a text-based web browser==. Our prototype copies how humans research answers to questions online—it submits search queries, follows links, and scrolls up and down web pages. ==It is trained to cite its sources==, which makes it easier to give feedback to improve factual accuracy. We’re excited about developing more truthful AI,[1](https://openai.com/research/webgpt#fn-1) but challenges remain, such as coping with unfamiliar types of questions.
>Language models like GPT-3 are useful for many different tasks, but have a tendency to “hallucinate” information when performing tasks requiring obscure real-world knowledge.[2](https://openai.com/research/webgpt#fn-2),[3](https://openai.com/research/webgpt#fn-3) To address this, we taught GPT-3 to use a text-based web-browser. ==The model is provided with an open-ended question and a summary of the browser state, and must issue commands== such as “Search ...”, “Find in page: ...” or “Quote: …”. In this way, the model collects passages from web pages, and ==then uses these to compose an answer==.


