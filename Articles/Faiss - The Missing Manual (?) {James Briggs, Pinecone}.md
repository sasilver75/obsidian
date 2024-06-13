Link: https://www.pinecone.io/learn/series/faiss/

In the past few years, vector search exploded in popularity; yet it was only with [[FAISS]] that this technology became more accessible!

## Chapter 1: Introduction to FAISS
[Video: James Briggs FAISS Introduction to Similarity Search](https://youtu.be/sKyvsdEv6rk)
- [[FAISS]] is a library developed by Facebook AI that enables efficient similarity search.
- Given a set of vectors, we can index them using FAISS -- then, using another vector (the query vector), we search for the most similar vectors within the index.

The first thing we need is some data! Let's download the data and extract the relevant columns into a single list:

```python
import requests
from io import StringIO
import pandas as pd

res = requests.get('https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt')
# create dataframe
data = pd.read_csv(StringIO(res.text), sep='\t')

# We take all samples from both sentence A and B, for a total of 14.5k unique sentences.
sentences = data["sentence_A"].tolist().extend(data["sentence_B"])

# Some filtering of NaNs from data
sentences = [word for word in sentences if type(word) is str]
# Now let's turn the sentences into vector representations
from sentence transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model.encode(sentences)

sentence_embeddings.shape #(14504, 768)
```

Once we FAISS installed
- Linux `conda install -c pytorch faiss-gpu` (GPU-optimized for CUDA-enabled Linux machines))
- MacOS/Windows `conda install -c pytorch faiss-cpu` (Slower)

We can build our first simple index with `IndexFlatL2`:

### IndexFlatL2
- ==IndexFlatL2== measures the L2 (Euclidean) distance between *all* given points between our query vector, and the vectors loaded into the index. It's simple, very accurate, but not too fast:
- ![[Pasted image 20240612235602.png|300]]
- In Python, we can initialize our IndexFlatL2 index with our vector dimensionality (768) like so:
```python
import faiss

d = sentence_embeddings.shape[1] # 768
index = faiss.IndexFlatL2(d)
index.is_trained # True
```
- The is_trained flag denotes whether the index is trained or not. Our index is not one that requires training.

Now we can load our embeddings and query like so:
```python
index.add(sentence_embeddings)
index.ntotal # 14504

# Let's do a query!
k = 4 # number of neighbors to return
xq = model.encode(["Someone sprints with a football"])
%%time
D, I = index.search(xq, k)
print(I) 
# [[4586 10252 12465  190]]
# CPU times: user 27.9 ms, sys: 29.5 ms, total: 57.4 ms
# Wall time: 28.9 ms

# We retrieved the row numbers; let's now select that subset of our dataframe using the iloc function.
data['sentence_A'].iloc[[4586, 10252, 12465, 190]]

# 4586    A group of football players is running in the field
# 10252    A group of people playing football is running past the person
# 12465      Two groups of people are playing football
# 190    A football player is running past an official

```
We were able to retrieve the top `k` vectors closest to our query vector!
If we wanted to extract the numerical vectors from FAISS, we can do that too:
```python
# We have k=4 vectors to return, so we initialize a zero array to hold them:
vecs = np.zeros((k,d))
# Then we can iterate through each ID from I, and add the reconstructed vector to our zero-array:
for i, val i nenumerate(I[0].tolist()):
	vecs[i, :] = index.reconstruct(val)

vecs.shape #(4, 768)
```

Using the `IndexFlatL2` index alone is computationally expensive; it doesn't scale well!
- When we use this index, we're performing an *==exhaustive search==*, meaning we're comparing our query vector `xq` to *every other vector in our index*; in this case, that's 14.5K L2-distance calculations every search!

![[Pasted image 20240613001313.png|300]]

We need to do something different, if we don't want our index to become too slow as we scale the number of documents! 

### Partitioning the Index
- A popular approach to optimize our search is to partition the index into Voronoi cells
![[Pasted image 20240613094610.png]]
- Using this method, we take a query vector `xq`, we *identify the cell it belongs to*, and *then* use our `IndexFlat2` (or another metric) to search between the query vector and all other vectors belonging to that specific cell.
	- Using this sort of hierarchical search means that we're producing an *approximate* answer, rather than an exact answer.

```python
# How many voroni cells we want in our index
nlist = 50 

# We use our L2 index as a quantizer step, which we then feed into the partitioning IndexIVFFlat index:
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
```
Now we have to train our index on our data (which we must do *before* adding any data to the index) 
```python
index.is_trained # False
index.train(sentence_dembeddings)
index.is_trained # True
# NOW that the index is trained, we can add the data.
index.add(sentence_embeddings)
index.ntotal  # 14504

# Let's now try searching
%%time
D, I = index.search(xq, k)
print(I)
# [[ 7460 10940  3781  5747]]
# CPU times: user 3.83 ms, sys: 3.25 ms, total: 7.08 ms
# Wall time: 2.15 ms

# Our search time clearly decreased, and in this case we don't find any difference between the results returned by our exhaustive search and this approximate search.
```

If it were the case that our approximate search with `IndexIVFFlat` returns suboptimal results,




## Chapter 2: Nearest Neighbor Indices for Similarity Search

## Chapter 3: Locality Sensitive Hashing (LSH): The Illustrated Guide

## Chapter 4: Random Projection for LSH

## Chapter 5: Product Quantization

## Chapter 6: Hierarchical Navigable Small Worlds (HNSW)

## Chapter 7: Composite Indices and the FAISS Index Factory



