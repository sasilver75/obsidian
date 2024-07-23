https://magazine.sebastianraschka.com/p/instruction-pretraining-llms?utm_campaign=email-half-post&r=764e6&utm_source=substack&utm_medium=email

---

In the last month, there have been a few interesting announcements:
- Apple integration of on-device LLMs
- Nvidia's [[Nemotron-4]] model family
- [[FlashAttention]] 3
- Google's [[Gemma 2]]
- More!

Let's cover some stuff


## Generating an Instruction Dataset from Nothing (Magpie)
- The [[Magpie]]: *Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing* paper highlights a practical exploit that seems super useful.
![[Pasted image 20240722232225.png|400]]
- As shown above, we just prompt *LLaMA 3 8B Instruct* with a pre-query template, and it generates an instruction for us. Then we feed the instruction back into the LLM, and it will generate a response. We do this a few thousands of times to get a dataset for instruction finetuning!
- It's strange that, with the resulting dataset, ==authors finetuned a LLaMA 3 8B *base* model with just IFT (no preference tuning) and found that it beats the original LLaMA 2 8B Instruct model ==! 
![[Pasted image 20240722234604.png|500]]
See that Magpie-Pro (distilled using Magpie from LLaMA 3 70B, finetuned LLAMA 3 8B) handily beats the LLaMA 3 Instruct.

The Magpie results shown in the figure above wer achieved with ==300 thousand samples only. In comparison, The original Llama 3 Instruct model was finetuned and aligned on 100 million samples!==

Sebastian has a [notebook here](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/05_dataset-generation/llama3-ollama.ipynb) implementing the generation process.

Authors created a "Pro" and "Air" version of the dataset, using either a 70B or 8B LLaMA Instruct model. The Pro dataset results in slightly stronger models compared to Air.

![[Pasted image 20240723003222.png|400]]
Interesting to see that the ratios are pretty similar for both.

Furthermore, the paper contains an analysis showing that the ==breadth or diversity in this dataset is much larger than that of other popular datasets for instruction finetuning==, such as Alpaca, Evol Instruct, and UltraChat.









