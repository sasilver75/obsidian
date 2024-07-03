https://huggingface.co/learn/nlp-course/chapter5/1?fw=pt

# Introduction
- We'll look at what datasets can do, and answer:
	- What do we do if a dataset isn't on the hub?
	- How do we slice and dice a dataset -- what if we need to use Pandas?
	- What do we do when the dataset is huge and will melt your laptop's RAM?
	- What is "memory mapping" and Apache Arrow?
	- How can you create your own dataset and push it to the Hub?

# What if my dataset isn't on the Hub?
- Datasets provides loading scripts to handle the loading of both local and remote datasets, supporting CSV&TSV, Text, JSON&JSOn Lines, Pickled Dataframes.
```python
# Wej ust need to specify the type of data, along with a path to it.
load_dataset("csv", data_files="my_file.csv")
load_dataset("pandas", data_files="my_dataframe.pkl")  # etc
```

Let's look at the [[SQuAD]]-it dataset, which is a large-scale dataset for QA in italian.
The training and test splits are hosted on Github, so we can download them all with a `wget` command
- `wget` is a linux utility used for retrieving files using HTTP/HTTPS/FTP/FTPS

```python
# Download
!wget https://github.com/crux82/squad-it/raw/master/SQuAD-it-test.json.gz
!wget https://github.com/crux82/squad-it/raw/master/SQuAD-it-train.json.gz

# Decompress the gzipped files
~gzip -dkv SQuAD_it-*.json.gz
# Teh compressed files have now been replaced with SquAD_it-test.json and SQuAD_it-train.json

# We can load a JSON file with the load_dataset() function, we just need to know whether we're dealing with ordinary JSON or JSON Lines (line-separated JSON).
	# In our JSON, all of the text is stored in a .data field
from datasets import load_dataset
squad_it_dataset = load_dataseT("json", data_files="SQuAD_it-train.json", field="data")

# By default, this creates a DatasetDict object with a train split:
squad_it_dataset
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
})

squad_it_dataset["train"][0]
{
    "title": "Terremoto del Sichuan del 2008",
    "paragraphs": [
        {
            "context": "Il terremoto del Sichuan del 2008 o il terremoto...",
            "qas": [
                {
                    "answers": [{"answer_start": 29, "text": "2008"}],
                    "id": "56cdca7862d2951400fa6826",
                    "question": "In quale anno si è verificato il terremoto nel Sichuan?",
                },
                ...
            ],
        },
        ...
    ],
}

# Great, we've got a local dataset. 
# But what we really want is o inclcdue BOTH the train and test splits into a single DatasetDict object so that we can apply our lovely Dataset.map() functions across both splits at once!
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}


squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# The loading scripts in HF Datasets actually suports automatic decompression of GZIP files, so we could have skipped the use of gzip by just pointing the data_files argument directly to the compressed files.
squad_it_dataset

DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
    test: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 48
    })
})

# Now we can apply various preprocessing techniques to clean up the data, tokenize the reviews, and so on. 

# Loading a Remote Dataset: Your dataset might be stored on a remote server! We just point the data_files argument to one or more URLs where the remote files are stored (rather than providing a file path like before)!
# For example, the SQuAD-it dataset was hosten on GitHub, so we can just point the data_Files to the SQuAD_it-*json.gz URLs as follows:
url = "https://github.com/crux82/squad-it/raw/master"
data_files = {
	"train": url+"SQuAD_it-train.json.gz",
	"test": url+"SQuAD_it-test.json.gz"
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# This returns the same DatasetDict object obtained above.
```

# Time to Slice and Dice
- Most of the time, your dataset won't be perfectly prepared for training models. Let's go over some of the various features that HF Datasets provides to clean up dataset!
- Datasets provides several functions to manipulate the contents of Dataset and DatasetDict objects.
	- We already encountered Dataset.map 

Let's cover the Drug Review Dataset hosted on UC Irvine Machine Learning Repository:
First, let's download and extract the data:
```python
!wget "https://archive.ics.uci.edu/ml/machinel-learning-databases/00462/drusCom_raw.zip"
!unzip drugsCom_raw.zip

# The file is TSV, which is just a variant of CSV, which uses tabs instead of commas as the seprator
from datasets import load_dataset
data_files = {"train", "drugsComTRain_raw.tsv", "test": "drusComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_Files, delimiter="\t")  # \t is tab is Python

# It's always good to take a random sample of your data to get a quick feel for the typeof data you're working with.
	# In DataSets, we can use a combination of .shuffle() and .select() together
drug_sample = drug_dataset["Train"].shuffle(seed=42).select(range(1000))  # Select a random 10000 from our training split; seed for reproducibility
drug_sample[:3]  # pull the first 3 of the split sample

{'Unnamed: 0': [87571, 178045, 80482],
 'drugName': ['Naproxen', 'Duloxetine', 'Mobic'],
 'condition': ['Gout, Acute', 'ibromyalgia', 'Inflammatory Conditions'],
 'review': ['"like the previous person mention, I&#039;m a strong believer of aleve, it works faster for my gout than the prescription meds I take. No more going to the doctor for refills.....Aleve works!"',
  '"I have taken Cymbalta for about a year and a half for fibromyalgia pain. It is great\r\nas a pain reducer and an anti-depressant, however, the side effects outweighed \r\nany benefit I got from it. I had trouble with restlessness, being tired constantly,\r\ndizziness, dry mouth, numbness and tingling in my feet, and horrible sweating. I am\r\nbeing weaned off of it now. Went from 60 mg to 30mg and now to 15 mg. I will be\r\noff completely in about a week. The fibro pain is coming back, but I would rather deal with it than the side effects."',
  '"I have been taking Mobic for over a year with no side effects other than an elevated blood pressure.  I had severe knee and ankle pain which completely went away after taking Mobic.  I attempted to stop the medication however pain returned after a few days."'],
 'rating': [9.0, 3.0, 10.0],
 'date': ['September 2, 2015', 'November 7, 2011', 'June 5, 2013'],
 'usefulCount': [36, 13, 128]
}
# see that the value for every key is a list

# Above: We can see a few quirks! 
	# There's an "Unnamed: 0" column that looks sort of like an anonymized ID or each patient
	# The condition column includes a mix of uppercase and lowercase labels
	# The reviews are of varying length, and contain a mix of Python line separators (\r\n) as well as HTML characters codes like &\#039;

# Let's see how we can use Datasets to deal with these issues!

# Let's confirm that the pateint ID hypothesis for the Unnamed:0 column, we can use the Dataset.unique() function to verify that the number of ID matches the number of rows in each split
for split in drug_dataset.keys():
	assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))

# Let's rename the Unnamed: 0 column to something a bit more interpretable
drug_dataset = drug_dataset.rename_column(  # This renames the Unnamed: 0 column for both the train and test splits!
	origina_column_name="Unnamed: 0", new_column_name="patient_id"
) # (Though TBH this makes me wonder what happens if/when the splits dont have the same column names?)

# Let's normalize all the condition labels labels (eg to lowercase) using a Dataset.map function, which applies a function to every "row" of the dataset
def lowercase_condition(example):
	return {"condition": example["conditions"]].lower()}  # Here, we're "rewriting over" an existing column, but we can also specify a new column!

drug_dataset.map(lowercase_condition)  # AttributeError: 'NoneType' object has no attribute 'lower'

# We've ran into a problem with our map function! From the error we can infer that some of the entries in teh condition column are none, and cannot be lowercased as they're not strings! Let's drop these rows, using Dataset.filter()
def filter_nones(example):
	return x["condition"] is not None

# Let's apply this same logic as a lambda function filter
drug_dataset = drug_dataset.filter(lambda example: example["condition"] is not None)

# With the Nones removed, we can now normalize our `condition` column:
drug_dataset = drug_dataset.map(lowercase_conditoin)

drug_dataset["train"]["condition"][:3]  # Just to see if the lowercasing worked
['left ventricular dysfunction', 'adhd', 'birth control']


# Creating new columns: Let's check the number of words in each review! First, let's define a simple function, and then then use Dataset.map!
def compute_review_length(example):
	return {"review_length": len(example["review"].split())}

# Now apply it with map (We can also use the Dataset.add_column() function, which IMHO is probably more explicit and nice)
drug_dataset = drug_dataset.map(compute_review_length)
drug_dataset["train"][0]
{'patient_id': 206461,
 'drugName': 'Valsartan',
 'condition': 'left ventricular dysfunction',
 'review': '"It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil"',
 'rating': 9.0,
 'date': 'May 20, 2012',
 'usefulCount': 27,
 'review_length': 17}
# As expected, we can see a review_length column has been added to our dataset!

# Let's use Dataset.filter() now to remove reviews that contain FEWER than 30 words!
drug_dataset = drug_dataset.filter(lambda example: example["review_length"] > 30)  # Keep the records for which this predicate evaluates to true
print(drug_dataset)
{'train': 138514, 'test': 46108}

# We can use the Dataset.sort() function to inspect the reviews with the largest numbers of words -- see the documentation to see which argument you need to use sort the reviews by length in descending order.

# Lastly, let's deal with the presenceof HTML character codes in our reviews, using Python's html modeul to UNESCAPE these characters:
import html
text = "I&#039;m a transformer called BERT"
html.unescape(text) # I'm a transformer called BERT"

# Let's use Dataset.map to unescape ALL the HTML characters in our corpus!
drug_dataset = drug_dataset.map(lambda example: html.unescape(example["review"]))  # map is quite useful for proecessing dat!

```

#### the map() method's superpowers
- takes a `batched` argument that, if set to True, causes it to send a batch of examples to the map function at once (configurable, but defaults to 1000)
	- When batched=True, the function receives a dictionary with the fields of the dataset, but each value is now a LIST OF VALUES, not just a single value.
	- The return value of Dataset.map() should be the same; a dictionary with the fields we want to *update or add to our dataset*.

Here's another way, for example, of unescaping all HTML characters, but using batched=True
```python
new_drug_dataset = drug_dataset.map(
	lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
) # This is going to execute MUCH faster than the previous one, and it's because list comprehensions are usually faster htan executing the same code in a for loop, and we gain performance by accessing lots of elements at the same time, instead of one by one.
	# Using Dataset.map with batched=True is essential to unlock the speed of the "fast" tokenizers that we'll encounter

# For instance, to tokenize all drug reviews with a fast tokenizer, we could use a function like this:
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
	return tokenizer(examples["review"], truncation=True)


# We can pass one or several examples to the tokenizer, so we can use this function with or without batched=True.
tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)

# We can tiem a single line by adding %time at the beginning of the line (%time print("hi")), or by adding %%time at the beginning of the cell on its own line.
# We see that using a fast tokenizer with batched=True is 30 times faster than its slow counterpart with no batching.

slow_tokenizer = AutoTokenizer.from_pretrained("bert-baes-cased", use_fast=False)  # ooh, cool "use_false" argument means to not use the faster rust-based tokenizer

def slow_tokenize_function(examples):
	return slow_tokenizer(examples["review"], truncation=True)

tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)
# In general, we don't recommend using Python multiprocessing for fast tokenizers with batched=True!!!

# But there's more! With Dataset.map and batched=True, you can change the number of elements in your dataset! This is super useful in many situations where you want to create several training features from one example.
# Let's tokenize oru examples and truncate them to a max length of 128, but ask the tokenizer to return ALL the chunks of the text, instead of just the first one!
def tokenize_and_split(example):
	return tokenizer(
		examples=["review"],  #? Can't find this
		truncation=True,
		max_length=128,
		return_overflowing_tokens=True
	)

# Let's test this on one example:
result = tokenize_an_split(drug_dataset["train"[0]])
[len(inp) for inp in result["input_ids"]]
[128, 49]  # So our first example in the training set became TWO FEATURES because it was tokenize to more than teh maximum number of tokens we specified!

# Let's do it for all elements
tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)  # ArrowInvalid: Column 1 named condition expected length 1463 but got length 1000

# Oh no, that didn't work! But why not? Looking at the error message, there's a mismatch in the lengths of one of the columns; one being length 1463 and the other length 1,000. 
	# Our 1,000 examples gave 1,463 new features, resulting in shape error (?)
	# The problem is that we're trying to mix two different datasets of different sizes. The drug_dataset columns have a certain number of examples, but the tokenized_dataset we're building will have more (1463; more than 1,000, because we're tokenizing long reviews into more than one example by using our return_overflowing_tokens=True argument.)
	# This doesn't work for a Datasaet, so we need to either remove the columns from the old dataset or make them the same size as they are in the new dataset.

# Let's do the former (removing the cols from the old dataset) with the remove_columns argument
tokenized_dataset = drug_dataset.map(
	 tokenize_and_split, batched=True, removed_columns=drug_dataset["train"].column_names
)

# Wem entioned that we can also deal with the mismatched length problem by making the old columns the same size as the new ones.
# To do that, we need the overflow_to-sample_mapping field that the tokenizer returns when we set return_overflowing_tokens=True
	# This gives us the mapping from a new feature index to the index of the sample it originated from. 
	# Using this, we can associate each key present in our original dataset with a list of values of the right size by repeating the values of each example as many times as it generates new features:
def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result

tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)  # Works!

# We've seen how Datasets can be used ot process data in various ways -- Datasts is designed to be interoperable with libraries like Pandas, NnumPy, PyTorch, TensorFlow, and JAX.

# To enable conversion between various third-party libraries, Datasets provides a Dataset.set_format() function.
# This function changes the OUTPUT FORMAT of the dataset, so you can easily switch to another format without affecting the underlying _data format_, which is Apache Arrow.
drug_dataset.set_format("pandas")
# Now when we access elements of the dataset we get a pd.Dataframe, instead of a dictionary!
drug_dataset["train"][:3]
```
![[Pasted image 20240620234200.png|300]]

```python
train_df = drug_dataset["train"][:]  # IIRC this is a cool way of making a copy, I think. But I think it also is a handy way of converting the dataset to a dataframe (because that's the set format) by iterating on the dataset

# From here, we can use all the PAndas functionality that we want; we can do fancy chaining to compute the class distribution among the condition entries. 
frequencies = (
   train_df["condition"].
   .value_countS()
   .to_frame()
   .reset_index()
   .rename(columns={"index": "condition", "condition": "frequency"})
)
frequencies.head()
```
![[Pasted image 20240620235230.png|250]]
Once we're done with our Pandas analysis, we can always  create a new Dataset object by using the Dataset.from_pandas() function 
```python
from datasets import Dataset

# We can turn our pandas dataframe back into a Dataset, if we like
freq_dataset = Dataset.from_pandas(frequencies)  # Remember (?) that Pd.dataframes are gonna be in-memory, iirc 🤔

# We can reset the output format of our drug_dadtaset from "pandas" to "arrow"
drug_dataset.reset_format()
```

Datasets also provides the ability to split our dataset into training and test sets, which is based on the famous functionality from scikit-learn
```python
drug_dataset_clean = drug_dataste["train"].train_test_split(train_size=0.8, seed=42)
# Let's rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# and add the "test" set to our DatasetDict
drug_dataset_clean["test"] = drug_dataset["test"]


# Finally, we can save our dataset to disk using .save_to_disk() for arrow, .to_csv() for csv, ._to_json() for JSON
	# Note for JSON and CSV formats, we have to store each split as a separate file
drug_dataset_clean.save_to_disk("drug-reviews")

# and later we can load from disk
from datasets import load_from_disk
drug_dataset_reloaded = load_from_disk("drug-reviews")
```
![[Pasted image 20240621002219.png|400]]

And then later we can load it from disk



# Big Data? Datasets to the rescue!
...


# Creating your own dataset


# Semantic search with FAISS


# Datasets, Check!


