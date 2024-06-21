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

# Above: We can see a few quirks! 
	# There's an "Unnamed: 0" column that looks sort of like an anonymized ID or each patient
	# The condition column includes a mix of uppercase and lowercase labels
	# 



```


# Big Data? Datasets to the rescue!


# Creating your own dataset


# Semantic search with FAISS


# Datasets, Check!


