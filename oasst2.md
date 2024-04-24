
Paper: April 14, 2023 -- Authors include Yannic Kilchner (lol), but I believe [[LAION]] is heavily involved.
It seems like this dataset was added ~8 months (~Dec 2023) after the publishing of the paper along with the [[oasst1]] dataset on HuggingFace.
Paper: [OpenAssistant Conversations -- Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327)
HuggingFace Dataset: [Open Assistant Conversations Dataset Release 2 (OASST 2)](https://huggingface.co/datasets/OpenAssistant/oasst2)

Dataset Summary
> This dataset contains message trees. Each message tree has an initial prompt message as the root node, which can have multiple child messages as replies, and these child messages can have multiple replies.
> 
> All messages have a role property: this can either be "assistant" or "prompter". The roles in conversation threads from prompt to leaf node strictly alternate between "prompter" and "assistant".
> 
> This version of the dataset contains data collected on the [open-assistant.io](https://open-assistant.io/) website until Nov 5 2023.