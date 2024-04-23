#article 
Link: https://huggingface.co/blog/synthetic-data-save-costs

----

Should you finetune your model or use an LLM API?
- Creating your model puts you in full control but requires expertise in data collection, training, and deployment. 
- LLM APIs are easier to use, but force you to send your data to a third party, and create costly dependencies on LLM providers.
	- What does a "breaking change" in LLM APIs really mean?

SO how can we combine the *convenience* of LLMs with the *control and efficiency* of customized models?

We'll use the case study of identifying investor sentiment in the news, and show how to use an open-source LLM to create synthetic data to train your customized model in a few steps.

Our resulting custom [[RoBERTa]] model can analyze a large new corpus for around $2.70, compared to $3061 with GPT!

![[Pasted image 20240422164530.png]]


## 1. The problem: There's no data for your use-case
- Imagine your boss asking you to build a sentiment analysis system for your company -- there are 100,000+ datasets on the HF Hub, 450~ of which have the "sentiment" word in the title.
- But if you work in (eg) a financial institution and you need to track sentiment towards the specific brands in your portfolio, none of the datasets are useful for your task.
- As a result, many people turn to general-purpose LLMs. These models are so large and general that they can tackle most tasks out of the box with impressive accuracy.
- Main disadvantage:
	- Size (General-purpose models have parameters that do things *other* than determine sentiment)
	- Control (You don't control these models)

## 2. The Solution: Synthetic data to teach efficient students
- In 20023, LLMs started reaching parity with human data annotators. ==There's now ample evidence showing that the best LLMs outperform crowd workers and are reaching parity with experts in creating quality (synthetic) data==.
	- It's hard to overstate the importance of this development!
	- The key bottleneck for creating tailored models was the money, time, and expertise required to recruit and coordinate human works to create tailored training data.
	- How, high-quality annotation labor is available through APIs; annotation instructions can be seen as prompts, and synthetic data is returned almost instantaneously with compute being the only bottleneck.

- ==In 2024, this approach will become commercially viable and boost the value of open-source for small and large businesses==.
	- Models (eg Mixtral-8x7B-Instruct-v0.1 by [[Mistral]]) now come with permissive software licenses.


## 3. Case study: Monitoring financial sentiment
- Imagine you're a developer in a large investment firm tasked with monitoring economic news sentiment toward companies in your investment portfolio... two options, until recently:
	1. You could fine-tune your own model, requiring writing annotation instructions, creating an annotation interface, recruiting crowdworkers, introducing QA measures to handle low-quality data, fine-tuning a model on this data, and deploying it.
	2. You could send your data with instructions to an LLM API; You skip the finetuning and deployment entirely, and you reduce the data analysis process to writing instructions (prompts) which you then send to an LLM annotator behind an API. In this case, the LLM API is your final inference solution.

Option 2 is more expensive at inference time and requires that you send sensitive data to a third party, but it's significantly easier to set up than Option 1, and, therefore, is used by many developers.

In 2024, synthetic data provides a third option:
- Combine the cost benefits of Option 1 with the ease-of-use of Option 2
- Use an LLM (the "teacher") to annotate a small sample of data for you, and then fine-tune a smaller, more efficient LM (the "student") on this data. This approach can be implemented in a few steps.
	- ((==I wonder==: Why have the teacher LM generate synthetic data, and then train a second smaller LM from scratch on that data, as opposed to fine-tuning the teacher LM on that data, and then directly *distilling* (ie training from the logits, not the dataset) it into a smaller model?))


### 3.1 Prompt an LLM to annotate your data
- We use the financial phrasebank sentiment dataset, a 3-class classification task, where 16 experts annotated sentences from financial news on Finnish companies as "positive", "negative", "neutral" from an investors' perspective.

We download some required packages
```python
!pip install datasets  # for loading the example dataset
!pip install huggingface_hub  # for secure token handling
!pip install requests  # for making API requests
!pip install scikit-learn  # for evaluation metrics
!pip install pandas  # for post-processing some data
!pip install tqdm  # for progress bars
```

Then we download the example dataset with its expert annotations
```python
from datasets import load_dataset

# We load the dataset from HuggingFace Hub
dataset = load_dataset("financial_phrasebank", "sentences_allagree", split='train')

# create a new column with the numeric label verbalised as label_text (e.g. "positive" instead of "0")
label_map = {
    i: label_text 
    for i, label_text in enumerate(dataset.features["label"].names)
}

def add_label_text(example):
    example["label_text"] = label_map[example["label"]]
    return example

dataset = dataset.map(add_label_text)

print(dataset)
# Dataset({
#    features: ['sentence', 'label', 'label_text'],
#    num_rows: 2264
#})

```

Now that we have our dataset, we write a short annotation instruction tailored to the `fniancial_phrasebank` task, and format it as an LLM prompt.

```python
prompt_financial_sentiment = """\
You are a highly qualified expert trained to annotate machine learning training data.

Your task is to analyze the sentiment in the TEXT below from an investor perspective and label it with only one the three labels:
positive, negative, or neutral.

Base your label decision only on the TEXT and do not speculate e.g. based on prior knowledge about a company. 

Do not provide any explanations and only respond with one of the labels as one word: negative, positive, or neutral

Examples:
Text: Operating profit increased, from EUR 7m to 9m compared to the previous reporting period.
Label: positive
Text: The company generated net sales of 11.3 million euro this year.
Label: neutral
Text: Profit before taxes decreased to EUR 14m, compared to EUR 19m in the previous period.	
Label: negative

Your TEXT to analyse:
TEXT: {text}
Label: """

```

Before we can pass this prompt to the API, we need to add some formatting to the prompt.
Most LLMs today are fine-tuned with specific chat template. This template consists of special tokens, enabling the LLM to distinguish between:
- The user's instructions
- System prompt
- Its own responses in a chat history

This template consists of special tokens, which enable LLMs to distinguish between the user's instructions, the system prompt, and its own responses in a chat history. 
Although we are not using the model as a chat bot here, omitting the chat template can still can still lead to silent performance degradation. ((I think this is just saying you should use the vocabulary that your model was trained on?))

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

chat_financial_sentiment = [{"role": "user", "content": prompt_financial_sentiment}]

prompt_financial_sentiment = tokenizer.apply_chat_template(chat_financial_sentiment, tokenize=False)

# The prompt now includes special tokens: '<s>[INST] You are a highly qualified expert ...  [/INST]'
```
The formatted annotation instruction (prompt) can now be passed to the LLM API.

We can log in to HF with the `huggingface_hub` library to safely handle our API token (or just define it as an environment variable)
```python
# you need a huggingface account and create a token here: https://huggingface.co/settings/tokens
# we can then safely call on the token with huggingface_hub.get_token()
import huggingface_hub
huggingface_hub.login()
```

We can then use the HF serverless API to run inference
```python
import os
import requests

# Choose your LLM annotator
# to find available LLMs see: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceClient.list_deployed_models
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

# docs on different parameters: https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
generation_params = dict(
    top_p=0.90,
    temperature=0.8,
    max_new_tokens=128,
    return_full_text=False,
    use_cache=False
)

# Package up our prompt with the generation parameters, send it off, and parse out the response
def generate_text(prompt=None, generation_params=None):
    payload = {
        "inputs": prompt, 
        "parameters": {**generation_params}
    }
    response = requests.post(
        API_URL, 
        headers={"Authorization": f"Bearer {huggingface_hub.get_token()}"}, 
        json=payload
    )
    return response.json()[0]["generated_text"]
```

As the LLM might not always return the labels in the same harmonized format.... we can define a short `clean_outpu` function, which maps the string output from the LLM to our three possible labels.
```python
labels = ["positive", "negative", "neutral"]

def clean_output(string, random_choice=True):
    for category in labels:
        if category.lower() in string.lower():
            return category
    # if the output string cannot be mapped to one of the categories, we either return "FAIL" or choose a random label
    if random_choice:
        return random.choice(labels)
    else:
        return "FAIL"

```
((==ABOVE==: This seems like a bad version of [[Instructor]]))


Now we can send our texts to the LLM for evaluation! 
We package our text into our prompt, send it (along with generation parameters) to the HF inference API, and parse the response.
```python
output_simple = []
for text in dataset:
	# add text into prompt template
	prompt_formatted = prompt_principal_sentiment.format(text=text)

	# send text to API
	output = generate_text(
		prompt=prompt_formatted,
		generation_params=generation_params
	)

	# clean output
	output_cl = clean_output(output, random_choice=True)
	# accumulate response
	output_simple.appeend(output_cl)

```

Based on this output, we can calculate metrics to see how accurately the model did the task, without pretraining on it.

Using the `sklearn` classification report to see how we did:
```python
from sklearn.metrics import classification_report

def compute_metrics(label_experts, label_pred):
    # classification report gives us both aggregate and per-class metrics 
    metrics_report = classification_report(
        label_experts, label_pred, digits=2, output_dict=True, zero_division='warn'
    )
    return metrics_report

label_experts = dataset["label_text"]
label_pred = output_simple

metrics = compute_metrics(label_experts, label_pred)
```

Based on the simple prompt, the LLM correctly classified (compared to the real label) 91.6% of texts -- that's pretty good, given that it was not trained to do this specific task.

We can improve this by using [[Chain of Thought]] and [[Self-Consistency]]
- [[Chain of Thought|CoT]] asks the model to reason about the correct label, and *then* make the labeling decision, rather than just immediately deciding on the correct label.