#article 
Link: https://huggingface.co/blog/synthetic-data-save-costs

It turns out that this was mostly a marketing blog post for their AutoTrain product, which is a finetuning-as-a-service tool from [[HuggingFace]] that lets you upload a CSV of data, select a model, and get a finetuned version of it automatically gets uploaded to your HuggingFace account.

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
		- ((This is actually what they do, later 😄))


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

# See this? This is where we stick in the special tokens
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
	output_simple.append(output_cl)

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
- [[Self-Consistency]] (SC) effectively gives the LLM multiple attempts per text, with different reasoning paths... and if the LLM then responds "positive" twice and "neutral" once, we choose the majority ("positive") as the correct label.

```python
prompt_financial_sentiment_cot = """\
You are a highly qualified expert trained to annotate machine learning training data.

Your task is to briefly analyze the sentiment in the TEXT below from an investor perspective and then label it with only one the three labels:
positive, negative, neutral.

Base your label decision only on the TEXT and do not speculate e.g. based on prior knowledge about a company. 

((THE CHAIN OF THOUGHT PROMPT))
You first reason step by step about the correct label and then return your label.

You ALWAYS respond only in the following JSON format: {{"reason": "...", "label": "..."}}
You only respond with one single JSON response. 

((NOTE THAT THEY HAVE THE RESPONSE AS INCLUDING A REASON AND LABEL KEY))
Examples:
Text: Operating profit increased, from EUR 7m to 9m compared to the previous reporting period.
JSON response: {{"reason": "An increase in operating profit is positive for investors", "label": "positive"}}
Text: The company generated net sales of 11.3 million euro this year.
JSON response: {{"reason": "The text only mentions financials without indication if they are better or worse than before", "label": "neutral"}}
Text: Profit before taxes decreased to EUR 14m, compared to EUR 19m in the previous period.	
JSON response: {{"reason": "A decrease in profit is negative for investors", "label": "negative"}}

Your TEXT to analyse:
TEXT: {text}
JSON response: """

# we apply the chat template like above
chat_financial_sentiment_cot = [{"role": "user", "content": prompt_financial_sentiment_cot}]
prompt_financial_sentiment_cot = tokenizer.apply_chat_template(chat_financial_sentiment_cot, tokenize=False)
# The prompt now includes special tokens: '<s>[INST] You are a highly qualified expert ...  [/INST]'
```

This is a JSON prompt where we ask the LLM to return a structured JSON string with its "reason" as one key and the "label" as another key.
The main advantage of JSON is that we can parse it to a Python dictionary and then extract the "label." We can also extract the "reason" if we want to understand the reasoning why the LLM chose this label.

((Again, this is a better use of the [[Instructor]] library, is my understanding -- just because you ask it to return some stuff in JSON doesn't mean it always will use the right keys, or the right datatypes when you try to json.load it.))

The `process_output_cot` function parses the JSON string returned by the LLM, and in the case where the LLM doesn't return valid JSON, tries to identify the label with a simple string match from our `clean_outpu` function defined above.

```python
import ast 

def process_output_cot(output):
    try: 
        output_dic = ast.literal_eval(output) 
        return output_dic
    except Exception as e:
        # if json/dict parse fails, do simple search for occurrence of first label term
        print(f"Parsing failed for output: {output}, Error: {e}")
        output_cl = clean_output(output, random_choice=False)
        output_dic = {"reason": "FAIL", "label": output_cl}
        return output_dic
```

We can now reuse our `generate_text` function from above with the new prompt, process the JSON Chain-of-Thought output with `process_output_cot` and send each prompt multiple times for Self-Consistency.

```python
self_consistency_iterations = 3

output_cot_multiple = []
for _ in range(self_consistency_iterations):
    output_lst_step = []
    for text in tqdm(dataset["sentence"]):
        prompt_formatted = prompt_financial_sentiment_cot.format(text=text)
        output = generate_text(
            prompt=prompt_formatted, generation_params=generation_params
        )
        output_dic = process_output_cot(output)
        output_lst_step.append(output_dic["label"])

    output_cot_multiple.append(output_lst_step)

# Now our output_cot_multiple is a list of lists, where each element is the response from an iteration of CoT propmting of a classification.
# We now just need some code to select the correct majority label:
import pandas as pd
from collections import Counter

# Given a collection of "votes" of class (From multiple Self-Consistent runs of CoT classification),
# select the majority option (or a random one, if there is no majority among the two labels)
def find_majority(row):
    # Count occurrences
    count = Counter(row)
    # Find majority
    majority = count.most_common(1)[0]
    # Check if it's a real majority or if all labels are equally frequent
    if majority[1] > 1:
        return majority[0]
    else: # in case all labels appear with equal frequency
        return random.choice(labels)

df_output = pd.DataFrame(data=output_cot_multiple).T

df_output['label_pred_cot_multiple'] = df_output.apply(find_majority, axis=1)

```

Now we can compare our improved LLM labels with the expert labels again and calculate metrics!

```python
label_experts = dataset["label_text"]
label_pred_cot_multiple = df_output['label_pred_cot_multiple']

# Let's now compute some metrics between our SC/CoT predicted labels and the real labels!
# We hope to beat the previous 91.6%... and we do, at 94%!
metrics_cot_multiple = compute_metrics(label_experts, label_pred_cot_multiple)

```

We boosted our 91.6% agreement (with ground truth labels) to 94% using these tricks, by giving the model time to think about its decision label (CoT) and by giving it multiple attempts (SC) (and by spending some additional compute).

Now we have a synthetic training dataset, thanks to these simple LLM API calls.

```python
df_train = pd.DataFrame({
    "text": dataset["sentence"],
    "labels": df_output['label_pred_cot_multiple']
})

df_train.to_csv("df_train.csv")
```


## Compare the Open-Source model to proprietary models
- How does the quality of synthetic data compare between Mistral's open-source Mixtral-8x7B-Instruct-v0.1 and OpenAI's GPT3.5 and GPT4?
- We ran the same pipeline and... we see that Mixtral performs better than GPT3.5 and is on par with GPT4 for this task!

### Understand Validate Validate our Synthetic data
- So far, the result is just some data annotated by a black-box LLM; but how can we trust the LLM annotations if we don't have expert annotations in a real-world scenario?
- ==We can only trust data that we've validated ourself==; Instructions/prompts always contain a degree of ambiguity, and even perfectly-intelligent annotators can make mistakes and must make unclear decisions when faced with often-ambiguous real-world data.

Fortunately, data validation has become significantly easier over the past years with open-source tools:
- [[Argilla]] provides a free interface for validating and cleaning *unstructured* LLM outputs.
- [[LabelStudio]] allows you to annotate data in many modalities
- [[CleanLab]] provides an interface for annotating data and automatically cleaning *structured* data.

It's essential to spend some time annotating texts to get a feel for the data and its ambiguities -- you'll quickly learn that the model made some mistakes, but there will also be several examples where the correct label is unclear, and some texts where you agree more with the LLM's decision than with the expert that created the dataset.

After less than an hour in the annotation interface, we get a better understanding of our data and corrected some of the ground-truth mistakes (or perhaps got some insights as to how to improve our model, based on the errors).


### 3.3 Tune your efficient and specialized model with AutoTrain
- So far, this has been a standard workflow of prompting an LLM through an API and validating the outputs.
- Now comes an additional step to enable significant resource savings: We fine-tune a smaller, but more efficient and specialized LM on the LLM's synthetic data. This process is called [[Distillation]], where the output from a larger model (the "teacher") is used to train a smaller model (the "student").
	- This just means that we take our original `text` from the dataset and treat the predictions from the LLM as our `labels` for fine-tuning.
	- ((It sounds like they're saying to train on the softmax classification labels, as opposed to the logits, which has a bunch more information))

We use the HuggingFace ==AutoTrain== solution to make this process even easier!
- AutoTrain is a no-code interface that enables you to upload a`.csv` file with labeled data, which the service then uses to finetune a model for you automatically!
- ((This is an automatic finetuning dataset from HuggingFace where you basically upload a .CSV and select a model architecture and get back a fine-tuned model.))

Training a small [[RoBERTa]] base model (130M parameters) on just 1811 data points is very fast and shouldn't take more than a few minutes! Once the training is done, ==the model gets automatically uploaded to your HuggingFace profile!==
- The whole process should take at most 15 minutes and cost less than $1!
- (If you want, you can even use the AutoTrain entirely local on your own hardware)

![[Pasted image 20240422192137.png]]

How well does our fine-tuned 0.13B parameter RoBERTa base model perform compared to much larger LLMs?
==It turns out that the custom model fine-tuned on 1811 texts achieves 94% accuracy -- the same as its teacher Mixtral and GPT4!== 
- A small model could never compete with a much larger LLM out of the box, but fine-tuning it on some high-quality data brings it to the same level of performance for the task that it's specialized in.

![[Pasted image 20240422192633.png]]

## Pros and Cons of different approaches

Three approaches
1. Manually creating your own data and model
2. Only using an LLM API
3. Using an LLM API to create synthetic data for a specialized model

What are their trade-offs?
![[Pasted image 20240422192715.png]]

# Conclusion
- We've shown the enormous benefits of using an LLM to create synthetic data to train a smaller, more effective model.