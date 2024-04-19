November 17, 2023
Paper: [Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2](https://huggingface.co/papers/2311.10702)
See previous work on evaluation of Instruction-Tuning datasets: [[Tulu]]

Releases:
- [[Tulu-v2-mix]], an improved collection of high-quality instruction datasets
- Tulu 2, a [[LLaMA 2]] finetune on the mixture above.
- Tulu 2+DPO, Tulu 2 trained using [[Direct Preference Optimization]], making it the largest (70B) model trained with DPO, at the time

(There were questions after the 7b [[Zephyr]] model was trained whether this fancy new [[Direct Preference Optimization|DPO]] technique would actually scale! This paper showed that it indeed did.)

Abstract
> Since the release of Tulu [Wang et al., 2023b], open resources for instruction tuning have developed quickly, from better base models to ==new finetuning techniques==. ==We test== and incorporate a number of these advances into Tulu, resulting in Tulu 2, a suite of improved Tulu models for advancing the understanding and best practices of adapting pretrained language models to downstream tasks and user preferences. Concretely, we release: (1) ==Tulu-V2-mix==, an ==improved collection of high-quality instruction datasets==; (2) ==Tulu 2==, ==LLAMA-2 models finetuned on the V2 mixture==; (3) ==Tulu 2+DPO==, Tulu 2 ==models trained with direct preference optimization (DPO),== including the largest DPO-trained model to date (Tulu 2+DPO 70B); (4) CODE Tulu 2, CODE LLAMA models finetuned on our V2 mix that outperform CODE LLAMA and its instruction-tuned variant, CODE LLAMA-Instruct. Our evaluation from multiple perspectives shows that the ==Tulu 2 suite achieves state-of-the-art performance among open models== and matches or exceeds the performance of GPT-3.5-turbo-0301 on several benchmarks. We release all the checkpoints, data, training and evaluation code to facilitate future open efforts on adapting large language models.
> 


![[Pasted image 20240418172026.png]]