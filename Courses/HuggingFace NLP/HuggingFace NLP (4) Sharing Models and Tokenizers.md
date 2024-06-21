https://huggingface.co/learn/nlp-course/chapter4/1?fw=pt

# The HuggingFace Hub
- The HuggingFace Hub is a central platform enabling discovery, use, contributions to new state of the art models and datasets.
	- > 400k models on on the Hub, not even limited to Transformers or NLP
- Sharing a model on the hub...
	- Means opening up to the community and making it accessible for people to use and fork.
	- Automatically deploys a hosted InferenceAPI for the model-- anyone can test it out directly on the model's page
	- Is free!

# Using Pretrained Models

If we wanted a french-based model that can do mask-filling, we can select the `caemembert-base` checkpoint to try it out!
- The identifier camembert-base is all we need to start using it! We can instantiate it using the `pipeline` function

```python
from transformers import pipeline

caemembert_fill_mask = pipeline("fill-task", model="camembert-base")  # task and checkpoint
results = camembert_fill-mask("Le camembert est <mask. :)")
[
  {'sequence': 'Le camembert est délicieux :)', 'score': 0.49091005325317383, 'token': 7200, 'token_str': 'délicieux'}, 
  {'sequence': 'Le camembert est excellent :)', 'score': 0.1055697426199913, 'token': 2183, 'token_str': 'excellent'}, 
  {'sequence': 'Le camembert est succulent :)', 'score': 0.03453313186764717, 'token': 26202, 'token_str': 'succulent'}, 
  {'sequence': 'Le camembert est meilleur :)', 'score': 0.0330314114689827, 'token': 528, 'token_str': 'meilleur'}, 
  {'sequence': 'Le camembert est parfait :)', 'score': 0.03007650189101696, 'token': 1654, 'token_str': 'parfait'}
]

# very simple!
# We can also instantiate the checkpoint using the model architecture directly!
from transformers import CamembertTokenizer, CamembertForMaskedLM  #??? TF?
tokenizer = CamembertTokenizer.frmo_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretreained("camembert-base")

# But we instead recommend using teh Auto* classes, since they're by design architecture-agnostic... makes switching checkpoints simple.
from transformers import AutoTokenizer, AutoModelforMaskedLM

checkpoint = "camembert-base"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(checkpoint)

# In the steps below, we take a look at the easiest way to share pretrained models to the hub.
# There are three ways to create new model repositories:
	# Using the push_to_hub API
	# Using the huggingface_hub Python library
	# USing the web interface

# Once you've created a repository, you can uplaod files to it via git and git-lfs. We'll walk you through creating model repositories and uploading files soon.

# First, you need to generate an authentication token so that HF knows who you are and to whose namespaces you have access to, etc.

# Logging in from a notebook:
from huggingface_hub import notebook_login
notebook_login()

# Logging in frmo a terminal:
huggingface-cli login
```
![[Pasted image 20240620202208.png|450]]

We can use the CLI to do various things
```bash
%% Downloading a single file %%
huggingface-cli download gpt2 config.json

%% Downloading an entire repository %%
huggingface-cli download HuggingFaceH4/zephyr-7b-beta

%% Downloading multiple files (sequentially) %%
huggingface-cli download gpt2 config.json model.safetensors

%% Downloading multiple files (providing patterns to filter, using --include and --exclude) %%
huggingface-cli download stabilityai/stable-diffisuion-xl-base-1.0 --include "*.safetensors" --exclude "*.fp16.*"

%% Downloading a datset or a Space %%
huggingflace-cli download HuggingFaceH4/ultrachat_200k --repo-type dataset
huggingface-cli download HuggingFaceH4/zephyr-chat --repo-type space

%% Downloading a specific revision %%
huggingface-cli download bigcode/the-stack --repo-type dataset --revision v1.1

%% Downloading to a local folder %%
%% The recommended and defaultw ay is to use the cache-system, but you can move them to a specific colder using the --local-dir option %%
huggingface-cli download adept/fuyu-8b model-00001-of-00002.safetensors --local-dir fuyu

%% By default, the cache directory is ~/.cache/huggingface, but you can change with HF_HOME env variable or using --cache-dir %%
huggingface-cli download gpt2 config.json --cache-dir ./path/to/cache

%% Specifying a token to access private repositories (by default, the token saved locally will be used, but you can also explicitly authn using --token %%
huggingface-cli download gp2 config.json --token=hf_****

%% You can pass the --quiet flag to not print out intermediate logs %%
%% You can change the default 10 second timeout (eg for slow connections) with HF_HUB_DOWNLOAD_TIMEOUT env var %%

%% huggingface-cli upload usage, to upload to the hub %%
%% huggingfae-cli upload [repo-id] [local_path] [path_in_repo] %%

%% Uploadindg an entire folder %%
huggingface-cli upload my-cool-model . .
%% To a specific destination on the repo %%
huggingface-cli upload my-cool-model ./path/to/curated/data /data/train

... Many more commands ...

```


# Sharing Pretrained Models
Anyways, more on sharing pretrained models.... continuing from above

```python
from transformers import TrainingArguments

training_Args = TrainingArguments(
	  "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True
)
# Now that we set push_to_hub=True, when we call trainer.train(), the trainer will then upload the model to the hub every time it is saved (here, every epoch) into a repository in your namespace! Repo will be named "bert-finetuned-mrpc", but you can choose a different name with hub_model_id="myorg/myreponame"

trainer = ...

trainer.train()

# Once you're done training, you should do a FINAL trainer.push_to_hub() to upload the last version of your model
trainer.push_to_hub()

# Accessing the model hub can be done directly on models, tokenizers, and configuration objects via their push_to_hub metho
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"

model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# If you belong to an organization, simply specify the organization's argument as (eg) organization="myorganization" too
	# Soemtiems you'll have to use the use_auth_token="<TOKEN>" if needed
model.push_to_hub("dummy_model")  # We can call push to hub on our model too :)
tokenizer.push_to_hub("dummy_model") # and the tokenizer! Note that we're using the same name...

# Along wit the CLI, we can use the huggingface_hub python library
from huggingface_hub import create_repo

# Other interesting parameters: private (to make the repo private), token (to override the token stored in your cache), repo_type (if you want to make a dataset or space instead of model)
create_repo("dummy-model", organization="huggingface") # This creates the dummy-model repo in the huggignface namespace, assuming you belong to that organization.

# We can also use the web interface, which is mostly self-explanatory
```

Going to skip some of this -- it's not super important yet. I'll take some sparse notes
- You can use git+git-lfs to sync a your repo on your machine to a repo on HuggingFace, so you have a more regular development flow.

# Building a Model Card
- A model card is a file which is arguably as important as the model and tokenizer files in a model repository. It's the central definition of the model, and ensures reusability and legibility by fellow community and organization members.
	- (It's the readme of a model)
- Creating the model card is done through the `README.md` file, which is just a markdown file.
	- Model description: Details about model (arch, version, author, general info, training procedures, parameters)
	- Intended uses and limitations (langauge, fields, domains)
	- How to use (might showcase usage of pipeline function/usage of model and tokenizer classes)
	- Limitations and bias 
	- Training data (description of datasets model trained on)
	- training procedure (relevant aspects of training from a reproducibility perspective (incl preprocessing and postprocessing of data), num epochs, bs, lr, etc.)
	- Evaluation results (How does model perform on evaluation dataset? Whats the decision threshold used in evaluation?)
	- variables and metrics (metrics used for evaluation, on which datasets+splits)
