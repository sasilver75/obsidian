December 7, 2023 (5 months after release of [[LLaMA 2]])
[[Meta AI Research]]
Paper: [LLaMA Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674)
#zotero 
Takeaway: LLaMA Guard is an input-output *safeguard model* (think: toxicity classifier based on a specially-instruction-tuned LLaMA 2 7B) that performs multi-class classification (and generates binary decision scores) on the "main" LM responses using a (replaceable) risk taxonomy+set of guidelines.

---
Deployed at Amazon Sagemaker, Together.AI, Databricks

Notes: 
- The LLaMA 2 responsible use guide recommends that products powered by Generative AI should deploy guardrails that mitigate all inputs and outputs to the models themselves, and have safeguards against generating high-risk or policy-violating content, as well as to protect against adversarial inputs and attempts at jailbreaking the model.
- This paper makes the following contributions:
	- Introducing a safety risk taxonomy related to interacting with AI agents.
	- LLaMA Guard, an LLM-based input-output safeguard model, finetuned on data labeled according to the above taxonomy. Allows users to customize the model input to adapt to other taxonomies.
		- The model *is* instruction-tuned using their specific (quite general) taxonomy, but it can zero-sho/few-shot transfer to a new taxonomy that you can provide in the prompt.
	- Different instructions for classifying human prompts vs AI responses; so LLaMA Guard is able to capture the semantic disparity between the user and agent roles.
	- Publicly release model weights, allowing practitioners to use the model without relying on paid APIs, as well as fine-tune the model.
- Safety Risk Taxonomy
	- A prerequisite to building systems is to have the following components:
		1. A **taxonomy of risks** that are of interest, which become the classes of the classifier.
		2. **Risk guidelines** that determine where the line is drawn between encouraged and discouraged outputs for each risk category in the taxonomy.
	- There's no one standard for these, so they've created a *sample taxonomy based on risk categories commonly considered in the mitigation of unintended outputs from large language models.*
		- ((Can I derive from this that they might likely be using a different taxonomy internally?))
	- Taxonomy:
		- **Violence and Hate** (discrimination, slurs, hateful sentiments)
		- **Sexual content** (statements *encouraging* sex acts, other explicit statements)
		- **Guns and Illegal Weapons** (statements encouraging/condoning planning of crimes using illegal acquisition or use of guns/illegal weapons (CBRN))
		- **Regulated/Controlled Substances** (produce/transfer/consume illegal drugs, tobacco, alcohol, cannabis)
		- **Suicide and Self Harm** (Condoning, encouraging, or enable self harm)
		- **Criminal Planning** (Miscellaneous statements that encourage/condone/help plan specific criminal activities like arson, kidnapping, or theft)
- Instruction-Following: For input-output safeguarding tasks, there are four key ingredients:
	1. **Set of guidelines**: Each task takes a set of guidelines as an input, consisting of numbered categories of violation and plaintext descriptions as to what is safe/unsafe in that category.
		- Although LLama Guard is instruction fine-tuned using their specific taxonomy, you can either fine-tune it further on different guidelines, or zero-shot/few-shot use a *new* policy without further finetuning.
	2. **Type of classification**: Each task indicates whether the model needs to classify the *user* messages (prompts) or the *agent* messages (responses). This is an important distinction, as these are two separate problems!
	3. **The conversation**: The conversation where users and agents take turn. May be single-turn or multi-turn.
	4. **The output format**: Each task specifies the desired output format, dictating the nature of the classification problem. In LLaMA Guard, the output contains two elements:
		- "safe" or "unsafe", both of which are *single tokens* in the [[SentencePiece]] tokenizer they use.
		- If the model outputs "unsafe," then the output should contain a new line, listing the taxonomy categories that are violated in the given piece of context.
		- LLaMA Guard accommodates binary and multi-label classification, where *the classifier score can be read from the probability of the first token.*


Abstract
> We introduce ==Llama Guard, an LLM-based input-output safeguard model geared towards Human-AI conversation use cases==. Our model incorporates a ==safety risk taxonomy==, a valuable tool for categorizing a specific set of safety risks found in LLM prompts (i.e., prompt classification). This taxonomy is also instrumental in classifying the responses generated by LLMs to these prompts, a process we refer to as response classification. For the purpose of both ==prompt== and ==response== ==classification==, we have meticulously gathered a dataset of high quality. Llama Guard, a ==Llama2-7b model that is instruction-tuned on our collected dataset==, albeit low in volume, demonstrates strong performance on existing benchmarks such as the OpenAI Moderation Evaluation dataset and ToxicChat, where its performance matches or exceeds that of currently available content moderation tools. Llama Guard functions as a language model, carrying out multi-class classification and generating binary decision scores. ==Furthermore, the instruction fine-tuning of Llama Guard allows for the customization of tasks and the adaptation of output formats. This feature enhances the model's capabilities, such as enabling the adjustment of taxonomy categories== to align with specific use cases, and facilitating zero-shot or few-shot prompting with diverse taxonomies at the input. We are making Llama Guard model weights available and we encourage researchers to further develop and adapt them to meet the evolving needs of the community for AI safety.

# Paper Figures

![[Pasted image 20240506162456.png|400]]
Above: The four main components of task instructions: Task type (user or agent), Taxonomy/Guidelines (rules), Conversation, and Output format (safe or unsafe+violations).


