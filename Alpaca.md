March 13, 2023
Blog: [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)

A 7B [[Decoder-Only Architecture]] language model based on [[LLaMA]] 7B, [[Instruction-Tuning|Instruction-Tuned]] using 52k instruction-following demonstrations generated from OpenAI's text-davinci-003 (~[[GPT-3.5]]). Doesn't have a commercial license, because LLaMA didn't. Started a mini-craze of open-source model fine-tuning. Trained at Stanford.

Abstract
> We introduce **==Alpaca 7B**, a model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations==. On our preliminary evaluation of single-turn instruction following, ==Alpaca behaves qualitatively similarly to OpenAI’s text-davinci-003, while being surprisingly small and easy/cheap to reproduce (<600$)==. Checkout our code release on [GitHub](https://github.com/tatsu-lab/stanford_alpaca).
> We emphasize that Alpaca is intended **only for academic research** and any **commercial use is prohibited**. There are three factors in this decision: First, Alpaca is based on LLaMA, which has a non-commercial [license](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform), so we necessarily inherit this decision. Second, the ==instruction data is based on OpenAI’s text-davinci-003==, whose [terms of use](https://openai.com/policies/terms-of-use) prohibit developing models that compete with OpenAI. Finally, we have not designed adequate safety measures, so Alpaca is not ready to be deployed for general use.

![[Pasted image 20240418165458.png]]



