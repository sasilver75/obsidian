July ~10, 2024
[[Microsoft Research]]
[Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena](https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/)
#zotero 
Takeaway: ...


Reference:
- Tweet: [Intro from a MSFT Wizard LM Person, mentions Wizard LM 3 is in the oven]( https://x.com/victorsungo/status/1811427047341776947)

-------

## Introduction
- Building an efficient data flywheel to continuously collect feedback to improve model capabilities is a key direction for next-generation AI research.
	- LMSYS's [[ChatBot Arena]] has been a significant development in this vein, which uses a diverse set of human evaluators to compare language models in head-to-head pairwise battle, using user/judge-supplied prompts. The preference data collected here has proven to be a valuable resource for model post-training and guidance, but human-based evaluations has some challenges:
		- It's time-consuming and resource-intensive, which limits the scale and frequency of evaluation.
		- Due to LMSYS's priority limitations, most models are unable to even participate in arena evaluations.



Abstract
> Recent work demonstrates that, post-training large language models with instruction following data have achieved colossal success. Simultaneously, human Chatbot Arena has emerged as one of the most reasonable benchmarks for model evaluation and developmental guidance. However, on the one hand, accurately selecting high-quality training sets from the constantly increasing amount of data relies heavily on intuitive experience and rough statistics. On the other hand, utilizing human annotation and evaluation of LLMs is both expensive and priority limited. To address the above challenges and build an efficient data flywheel for LLMs post-training, we propose a new method named ==Arena Learning==, by this way we can ==simulate iterative arena battles among various state-of-the-art models on a large scale of instruction data, subsequently leveraging the AI-anotated battle results to constantly enhance target model== in both supervised fine-tuning and reinforcement learning. For evaluation, we also introduce ==WizardArena==, which can ==efficiently predict accurate Elo rankings between different models based on a carefully constructed offline testset==, WizardArena ==aligns closely with the LMSYS Chatbot Arena rankings==. Experimental results demonstrate that our WizardLM-β trained with Arena Learning exhibit significant performance improvements during SFT, DPO, and PPO stages. This new fully AI-powered training and evaluation pipeline achieved 40x efficiency improvement of LLMs post-training data flywheel compare to LMSYS Chatbot Arena.


# Paper Figures
![[Pasted image 20240713135349.png|600]]
...

![[Pasted image 20240713135409.png|600]]
Given the same prompt, two models respond, and a Judge LM assigns scores to each and a rationale to determine which is better.
We can use the winning response for supervised fine-tuning, and the (winning, losing) pair for DPO or PPO preference training.


# Non-Paper Figures














