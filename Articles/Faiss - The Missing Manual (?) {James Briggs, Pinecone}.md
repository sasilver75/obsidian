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

If it were the case that our approximate search with `IndexIVFFlat` returns suboptimal results, we can improve accuracy by ***increasing the search scope***. We do this by increasing the `nprobe` attribute value, which defines *how many nearby cells to search!*
![[Pasted image 20240613095353.png|300]]

```python
index.nprobe = 10

$$time
D, I = index.search(xq, k)
print(I)

# [[ 7460 10940  3781  5747]]
# CPU times: user 5.29 ms, sys: 2.7 ms, total: 7.99 ms
# Wall time: 1.54 ms
```

![[Pasted image 20240613095616.png]]
Above: ((Confusing: The docs and code results say things like: "Because we're searching a larger scope by increasing the nprobe value, we will see the search speed increase too", and indeed my code results somehow show faster results by increasing nprobe, which doesn't really make sense to me. Confusingly, this chart seems to show the intuitive result, which is that probing for more documents results in slower search speeds. AHHH I'm looking at the wrong time! Don't look at the wall block time, look at the last number on the previous line.))

### Vector Reconstruction
- If we go and attempt to use our index.reconstruct(<vector_idx>) again, we'll get a RuntimeError, since there's no direct mapping between the original vectors and their index position, due to the addition of the IVF step.
- If we'd like to reconstruct the vectors, we must first create these direct mappings using index,make_direct_map():
```python
index.make_direct_map()
index.reconstruct(7640)[:100]
# (A vector output)
```

### Quantization
- There's another key optimization to cover; So far, all of our indices have stored our vectors as full (`Flat`) vectors., In large datasets, this requires a lot of memory for storage!
- FAISS comes with the ability to compress our vectors using ==Product Quantization (PQ)==
- We can view PQ as an additional approximation step, similar to our use of ==IVF (Inverted File Index).==
	- Where IVF allowed us to approximate by *reducing the scope* of our search, PR approximates the *distance/similarity calculation* instead!
	- PQ achieves this by compressing the vectors themselves, in three steps:
		- We split the original vector into several subvectors
		- For each set of subvectors, we perform a clustering operation, creating multiple centroids for each sub-vector set.
		- In our vector of sub-vectors, we replace each subvector with the ID of its nearest set-specific centroid.
![[Pasted image 20240613100823.png|400]]

To implement this, we use the IndexIVFPQ index (which also needs to be trained)

```python
m = 8  # n centroid IDs in final compressed vectors
bits = 8  # number of bits in each centroid

# We keep the asme L2 distance flat index
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

index.is_trained  # False
index.train(sentence_embeddings)
index.is_trained  # True
index.add(sentence_embeddings)

index.nprobe = 10  # Align to the previous IndexIVFFlat nprobe value

%% time
D, I = index.search(xq, k)
print(I)
# [[ 5013 10940  7460  5370]]
# CPU times: user 3.04 ms, sys: 2.18 ms, total: 5.22 ms
# Wall time: 1.33 ms
```

### Speed or Accuracy?
- By adding PQ, we've reduced our IVF search time from ~7.5ms to ~5ms, a small difference on a dataset this size -- but when scaled up, this becomes significant quickly!
	- Note that we also ended up getting slightly different results. BOTH of our speed operations (==IVF== and ==PQ==) come at the cost of accuracy.

![[Pasted image 20240613102047.png|350]]

## Chapter 2: Nearest Neighbor Indices for Similarity Search
- One of the key components to efficient search is flexibility; there's no one-size-fits-all in similarity search! So how do we choose an index? Should we have multiple?
	- We'll explore the pros and cons of some of the most important indices:
		- Flat
		- LSH
		- HNSW
		- IVF

Similarity search is useful to compare data quickly; given a query in any format (text, audio, video), we can use similarity search to return relevant results.

Many indexing solution are available; one in particular is called FAISS, which comes with many different index types (many of which can be mixed and matched to produce multiple layers of indices).
- Which index to use depends on our use case; we consider factors like dataset size, search frequency, or search quality vs. search speed.

### Flat Indexes
- Flat indices are the simplest; ==indexes are *flat* because we don't modify the vectors that we feed into them==.
- Because there's no approximation or clustering of our vectors, ==flat indices produce the most accurate results,== but the perfect search quality comes at the cost of ==significant search times.==
![[Pasted image 20240613103504.png|350]]
After calculating all of these distances, we will return the k best matches.

So when should we use a flat index?
- When search quality is unquestionably high priority, and search speed is less important.
	- This can be the case for smaller datasets, or when you're using more powerful hardware.

Implementing a flat index in FAISS can be done with one of the two flat indexes:
1. `IndexFlatL2` if using euclidean/***L2*** distance
2. `IndexFlatIP` if using ***inner product*** distance (slightly faster)

![[Pasted image 20240613103915.png|200]]


```python
d = 128  # dimensionality of our Sift1M dataset data
k = 10  # number of nearest neighbors to return

index = faiss.IndexFlatIP(d)
index.add(data)
D, I = index.search(xq, k)
```
Above:  See that our flat indexes don't need training (as we have no parameters to optimize when storing vectors without transformations or clustering)

Flat indexes are brilliantly accurate, but terribly slow. In similarity search, there's a trade-off between search speed and search quality.

How can we make our search faster?
1. ==Reduce vector size==: Through dimensionality reduction or reducing the number of bits representing our vectors values.
2. ==Reduce search scope==: By clustering or organizing vectors into tree structures based on certain attributes, similarity, or distance -- and then restricting our search to closest clusters or filtering through most similar branches.


### Locality-Sensitive Hashing (LSH)
- In [[Locality Sensitive Hashing]] (LSH), a wide range of performances are heavily dependent on the parameters set. Good quality results in slower search, and fast search results in worse quality. Poor performance for high-dimensional data.
- Locality Sensitive Hashing (LSH) ==works by grouping vectors into buckets by processing each vector through a hash function that maximizes hashing collisions -- rather than minimizing, as is usual with hashing functions==.
![[Pasted image 20240613104914.png|300]]
Python dictionaries are examples of hash tables using typical hashing functions that *minimize* hashing collisions -- for LSH, we instead want to *maximize* hashing conditions.
- Why would we want to maximize collisions? For LSH to group similar objects together... when we introduce a new query object (or vector), our LSH algorithm can be used to find the closest matching groups.
![[Pasted image 20240613105228.png|350]]

Implementing LSH in FAISS is easy:
```python
# Set resolution of bucketed vectors
# Higher values mean greater accuracy at cost of more memory and slower search speeds
nbits = d*4  

index = faiss.IndexLSH(d, nbits)
index.add(wb)

D, I = index.search(xq, k)
```
![[Pasted image 20240613105402.png|300]]
Our baseline `IndexFlatIP` is our 100% recall performance; using `IndexLSH`, we can achieve 90% using a very high `nbits` value.
- This could be a reasonable sacrifice if we get improved search times!
But ==LSH is highly sensitive to the curse of dimensionality when using a larger d value==, so we also need to increase `nbits` to maintain search quality.
- So our stored vectors become increasingly larger as our original vector dimensionality `d` increases. This quickly leads to excessive search times:
![[Pasted image 20240613105723.png|300]]
Which is mirrored by our index memory size:
![[Pasted image 20240613105753.png|300]]
==So `IndexLSH` is NOT SUITABLE if we have a large vector dimensionality; 128 is already *too large*; instead, LSH is suited to low-dimensionality vectors -- and small indices.==
- If we find ourselves with large `d` values or large indices, we avoid LSH completely, instead focusing on our next index, HNSW.

### Hierarchical Navigable Small Worlds Graphs (HNSW)
- [[Hierarchical Navigable Small Worlds]] (HNSW) indices consistently top out as the ==highest performing indices==. They're a further adaptation on Navigable Small World (NSW) graphs, which are graph structures containing vertices connected by edges to their nearest neighbors. The "NSW" part is due to vertices in the graph having very short average path lengths to all other vertices -- on Facebook, despite 1.59B active users, the average number of hops to traverse the friendship graph from one user to another was just 3.57.
	- HNSW graphs are built by taking NSW graphs and breaking them apart into multiple layers. With each incremental layer eliminating intermediate connections between vertices.
	- HNSQ gives great search quality, good search speed, but substantial index sizes.
- ![[Pasted image 20240613110500.png|300]]
- ==For bigger datasets, with higher-dimensionality, HNSW graphs are some of the best-performing indices we can use. By layering it with other quantization steps, we can improve search-times even further!==

To build it in FAISS, all we need is `IndexHNSWFlat`:
```python
# Set our HNSSW index parameters
M = 64 # number of connections each vertex will have
ef_search = 32 # depth of layers explored during search
ef_construction = 64 # depth of layers explored during index construction

index = faiss.IndexHNSWFlat(d, M)
# set our efConstruction and efSearch parameters
index.hnsw.efConstructoin = ef_construction
index.hnsw.efSearch = ef_search
index.add(wb)

# search as usual
D, I = index.search(wb, k)
```
- `M` — the number of nearest neighbors that each vertex will connect to.
- `efSearch` — how many entry points will be explored between layers during the search.
- `efConstruction` — how many entry points will be explored when building the index.
![[Pasted image 20240613110954.png|350]]
M and efSearch have a larger impact on search-time; efConstruction primarily increases index construction time (meaning slower index.add), but at higher M values and higher query volume we do see an impact from efConstruction on search time too.
![[Pasted image 20240613111223.png|300]]
HNSW gives us great search quality at very fast search speeds, but there's always a catch! HNSW indices take up a significant amount of memory; using an M value of 128 for the Sift1M dataset requires 1.6GB of memory!
![[Pasted image 20240613111809.png|300]]
We can increase our two other parameters efSearch and efConstruction with no effect on the index memory footprint.
- ==So where RAM isn't a limiting factor, HNSW is great as a well-balanced index that we can push to focus more towards quality by increasing our three parameters!==

### Inverted File Index
- ==Inverted File Index (IFV)== consists of search scope reduction through clustering! It's a very popular index, as it's easy to use, with high search quality and reasonable search speed, along with reasonable memory usage.
- IFV works on the concept of ==Voroni diagrams==, also called Dirchlet tesselation.

How do Voronoi diagrams work?
- We imagine our highly dimensional vectors in a 2D space, then place a few additional points in our 2D space, which become our cluster (Voronoi cell) centroids.
- We then extend an equal radius out from each of our centroids; at some point, the circumfrences of each cell collide with eachother, and create our cell edges:
![[Pasted image 20240613114031.png|300]]
Now every datapoint will be contained in a cell and be assigned to that respective centroid.
- There's a problem if our query vector lands near the edge of a cell; there's a good chance that its closest other datapoint is contained within a *neighboring cell!*

![[Pasted image 20240613114121.png|300]]

What we can do to mitigate this issue is to increase an index parameter known as the `nrpobe` value, which sets the number of cells to search.

![[Pasted image 20240613114202.png|300]]

Implementing [[Inverted File Index|IVF]] is straightforward, using `IndexIVFFlat`

```python
nlist = 128 # Number of cells/clusters to partition data into
quantizer = faiss.IndexFlatIP(d)  # How the vectors will be stored/compared
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(data)  # We need to train the index to cluster into cells
index.add(data)

index.nprobe = 8
D, I = index.search(xq, k)
```
- `nprobe` — the number of cells to search (Increase this to increase search scope, prioritizing search quality at the expense of speed)
- `nlist` — the number of cells to create (Increase this to prioritize search speed, since there will be fewer vectors in the retrieved cell(s))


## Chapter 3: Locality Sensitive Hashing (LSH): The Illustrated Guide
- [[Locality Sensitive Hashing]] (LSH) is one of the original techniques for producing high-quality search, while maintaining lightning fast sublinear search speeds.

- LSH consists of a variety of different methods; in this article, we'll cover the traditional approach, which consists of multiple steps:
	- Shingling
	- MinHashing
	- The final banded LSH function

- LSH allows us to segment and hash the same sample several times, and when we find that a pair of vectors have been hashed to the same value *at least once*, we tag them as *candidate pairs* (potential matches).
- LSH uses hash functions that try to *maximize colissions* (ideally only for *similar inputs*).
	- There's no single approach to hashing in LSH -- but they all share the same "bucket similar samples through a hash function" logic.

### Shingling, MinHashing, and LSH
- The LSH approach we're going to explore consists of a three-step process:
	1. Convert text to sparse vectors using a k-shingling (and one-hot encoding)
	2. Use minhashing to create "signatures"
	3. These signatures are passed onto our LSH process to weed out candidate pairs.
![[Pasted image 20240613114905.png|300]]

### ==k-Shingling== (shingling)
- The process of converting a string of text into a set of "shingles." 
- Similar to moving a window of length `k` down our string of text, and taking a picture at each step; we collate all of these pictures to create our set of shingles.
![[Pasted image 20240613120600.png|300]]
We move through a string and add k characters at a time to a 'shingle set'
- Shingling also removes duplicate items (hence the word 'set'). We can create a simple k-shingling function in Python:
```python
def shingle(text: str, k: int):
	shingle_set = set()
	for i in range(len(text) - k+1):
		shingle_set.add(text[i:i+k])
	return shingle_set
```

Now that we have our shingles, we create our sparse vectors! To do this, we just need to union all of our sets to create one big set containing *all* of the shingles across all of our sets -- we call this the ==vocabulary== (or vocab).

![[Pasted image 20240613123348.png|300]]

We use this vocab to create our sparse vector representations of each set; all we do is create an empty vector full of zeros, the same length as our vocab; then, we look at which shingles appear in our set.
- For every shingle that appears, we identify the position of that shingle in our vocab and set the respective position in our new zero-vector to 1. 

![[Pasted image 20240613123512.png|300]]

### Minhashing
- The next step in our process is [[MinHash]]ing; we want to convert our sparse vectors into dense vectors.
- We want to randomly generate one minhash function for every position in our signature (eg the dense vector). 
	- So if we wanted to create a dense vector/signature of 20 numbers, we'd use 20 minhash functions.
- We take a randomly-permuted count vector (from 1 to len(vocab)+1) and find the minimum number that aligns with 1 in our sparse vector. ([animation](https://d33wubrfki0l68.cloudfront.net/2b77a22ec9933902bb46e2f19b753654fc58d145/91c4c/images/locality-sensitive-hashing-8.mp4))
	- We look at our sparse vector and say: Did this shingle at `vocab[1]` exist in our set? If it did, the sparse vector value will be 1. Let's pretend that it's not 0; so we look to `vocab[2]` and ask the same question. If the answer is *yes*, then our minhash output is `2`.
- So this is how we produce one value in our minhash signature; but we need to produce 20 (or more) of these values; so we assign a different minhash function to each signature position, and repeat the process:
![[Pasted image 20240613125017.png|300]]
Above: I really have no idea how this is working.

At the end of this process, we produce our minhash signature (or dense vector).

In code, this might help:
```python
hash_ex = list(range(1, len(vocab)+1))  # [1, 2, 3, ...]

from random import shuffle

shuffle(hash_ex)  # [63, 7, 94, ...]

# Loop through this randomized MinHash vector (starting at 1) and match the index of each value to the equivalent values in the sparse vector (by index)

for i in range(1, 5):
	# list.index returns the index of the specified element in the list (else ValueError)
    print(f"{i} -> {hash_ex.index(i)}")

1 -> 58
2 -> 19
3 -> 96
4 -> 92

# What we do with this is count up from `1` to `len(vocab) + 1` and find if the resultant `hash_ex.index(i)` position in our one-hot encoded vectors contains a positive value (`1`) in that position, like so:
for i in range(1, len(vocab)+1):
	# Find the index of i in the shuffled list.
	idx = hash_ex.index(i)
	signature_val = a_1hot[idx]
	print(f"{i} -> {idx} -> {signature_val}")
	if signature_val == 1:
		print('match')
		break

1 -> 58 -> 0
2 -> 19 -> 0
3 -> 96 -> 0
4 -> 92 -> 0
5 -> 83 -> 0
6 -> 98 -> 1
match!

```

Next, we build a signatures from multiple iterations of 1 and 2 (we'll formalize the code from above into a few easier to use functions):

```python
def create_hash_func(size: int):
	# Creates the hash vector/function
	hash_ex = list(range(1, len(vocab)+1))
	shuffle(hash_ex)
	return hash_ex

def build_minhash_func(vocab_size: int, nbits: int):
	# Builds multiple minhash vectors
	hashes = []
	for _ in range(nbits):
		hashes.append(create_hash_func(vocab_size))
	return hashes

# We cerate 20 minhash vectors
minhash_func = build_minhash_func(len)vocab), 20)

def create_hash(vector: list):
	# Use this fn for creating our signatures (eg the matching)
	signature = []
	for func in minhash_func:
		for i in range(1, len(vocab) + 1):
			idx = func.index(i)
			signature_val = vector[idx]
			if signature_val == 1:
				signature.append(idx)
				break
	return signature

# Now we can create signatures
a_sig = create_hash(a_1hot)
# [44, 21, 73, 14, 2, 13, 62, 70, 17, 5, 12, 86, 21, 18, 10, 10, 86, 47, 17, 78]
```

And that's MinHashing! We've taken a sparse vector and compressed it into a more densely packed, 20-number signature.

but is information really maintained between our much larger sparse vector and much smaller dense vector? If so, surely the similarity between vectors will be similar too, right?

We can use [[Jaccard Similarity]] to calculate the similarity between our sentences in shingle format, then repeat this for vectors in signature format!

```python
def jaccard(a: set, b: set):
	return len(a.intersection(b)) / len(a.union(b))

jaccard(a, b), jaccard(set(a_sig), set(b_sig))
# (0.14814814814814814, 0.10344827586206896)
```
We see *pretty close* similarity scores for both, so it seems that some information is being retained.

### Band and Hash
- The final step in identifying similar sentences is the LSH function itself.
- We're taking the banding approach to LSH, which we could describe as the traditional method -- it involves taking our signatures, hashing segments of each signature, and looking for hash collisions.
- If we hash each of these dense vectors as whole, we might struggle to build a hashing function that accurately identifies similarity between them (we don't require that a full vector is equal, only that parts of it are similar).
	- We want signatures that share even *some* similarity to be hashed into the same bucket, thus being identified as candidate pairs!

The banding method solves this problem by splitting our dense vectors into sub-parts called *bands*. then, rather than processing the full vector through our hash function, we can pass *each band of our vector* through a hash function.

![[Pasted image 20240613132051.png|400]]

Now we can add a more flexible condition -- given a collision between *any two sub-vectors*, we can consider the full vectors as candidate pairs.

![[Pasted image 20240613132235.png|400]]
Now, only part of the two vectors must match for us to consider them as candidate pairs. This increases the number of false positives.

Implementation:
```python
def split_vector(signature, b):
	assert len(signature) % b == 0
	r = int(len(signature)/b)  # Length of subvector
	# code splitting signature in b parts
	subvecs = []
	for i in range(0, len(signature), r):
		subvecs.append(signature[i: i+r])
	return subvecs

band_a = split_vector(a_sig, 10) # [[90, 43], [69, 55], ...]
band_b = split_vector(b_sig, 10)

# We loop through the lists to identify any matches between sub-vectors; if we find ANY matches, we take them as candidate pairs

for b_rows, c_rows in zip(band_b, band_c):
	if b_rows == c_rows:
		print(f"Candidate pair: {b_rows} == {c_rows}")
		break
```

Testing LSH using FAISS
(assume we have a data variable, containing sentences)
```python
k = 8 # Shingle size

# build shingles (each sentence -> set of shingles)
shingles = [build_shingles(sentence, k) for sentence in sentences]

# build vocabulary (of all unique shingles in shingles)
vocab = build_vocab(shingles)

# One-hot encode our sentence shingles
shingles_1hot = []
for shingle_set in shingles:
	shingles_1hot.append(one_hot(shingle_set, vocab))

# Stack them into a single numpy array of 4500 sparse vectors, with each vector of length 36466
shingles_1hot = np.stack(shingles_1hot)
shingles_1hot.shape  # (4500, 36466)

# Now let's compress our sparse vectors into dense vector "signatures"
arr = minhash_arr(vocab, 100)

# Get our signatures
signatures = [get_signature(arr, vector) for vector in shingles_1hot]

# Merge signatures into a single array
signatures = np.stack(signatures)
# We've compressed our sparse vectors of length 36466 to signatures of length 100, while still retaining information!
signatures.shape  # (4500, 100)

signatures[0]
# array([  65,  438,  534, 1661, 1116,  200, ... ])

# Now, onto the LSH portion
b = 20
lsh = LSH(b)

for signature in signatures:
	lsh.add_hash(signature)

lsh.buckets


[{'65,438,534,1661,1116': [0],
  '65,2199,534,806,1481': [1],
  '312,331,534,1714,575': [2, 4],
  '941,331,534,466,75': [3],
  ...
  '5342,1310,335,566,211': [1443, 1444],
  '1365,722,3656,1857,1023': [1445],
  '393,858,2770,1799,772': [1446],
  ...}]
# Note that lsh.buckets contains a separate dictionary for each band -- we do not mix buckets between different bands.

# Now we just need to extract our candidate pairs by looping through buckets and extracting pairs.
candidate_pairs = lsh.check_candidates()
len(candidate_pairs)
# 7243

list(candidate_pairs)[:5]
# [(1646, 1687), (3234, 3247), (1763, 2235), (2470, 2622), (3877, 3878)]
```

Now that we've identified candidate pairs, we can restrict our similarity calculations to those pairs only! We'll find some that will be within our similarity threshold, and others that will not.
- The goal was to *restrict our scope*, reducing search complexity while still maintaining high accuracy in identifying pairs.
![[Pasted image 20240613134115.png|250]]

### Optimizing the bands
- It's possible to optimize our band value b to shift the similarity threshold of our LSH function! This similarity threshold is the point at which we'd like our LSH function to switch from a non-candidate to a candidate pair.

![[Pasted image 20240613134219.png]]

![[Pasted image 20240613134239.png]]

If we return more candidate pairs, we'll naturally have more false positives. This is an unavoidable consequence of modifying b.

## Chapter 4: Random Projection for LSH
- Recall: Searching with LSH consists of three steps:
	1. Index all of our vectors into their hashed vectors.
	2. Introduce our query vector (search term). It is hashed using the same LSH function.
	3. Compare our hashed query vector to all other hash buckets via Hamming distance — identifying the nearest.

- We should note that grouping vectors into lower-resolution *hashed vectors* means that our search isn't exhaustive (eg comparing *every vector*), so we should expect a lower search quality compared to flat indices.

The two most popular approaches to LSH are:
- Shingling, MinHashing, and banded LSH (traditional approach)
- Random hyperplanes with dot-product and Hamming distance

We covered the former in the last section, so let's cover the latter.

### Random Hyperplanes
- The random hyperplanes approach is deceptively simple - although it can be hard to find details on the method.
- ==Using the random projection method, we reduce our high-dimensioned vectors into low-dimensionality binary vectors; once we have these vectors, we can measure the distance between them using [[Hamming Distance.==
- The hyperplanes are used to split our datapoints and assign a value of 0 for those that appear on the negative side of the hyperplane and a 1 for those that appear on the positive side.
![[Pasted image 20240613140912.png|300]]
- To identify which side of the hyperplane our data point is located, all we need is the normal vector of the plane (eg a vector perpendicular to the plane). We feed this normal vector (alongside our datapoint vector) into a [[Dot Product]] function -- if the two vector share the same direction, the resultant dot product is positive, else negative.
- As we add *more hyperplanes,* the amount of encoded information rapidly increases! By projecting our vectors into lower-dimensional spaces using these hyperplanes and the manner described above, we produce new hashed vectors.
	- ((To me, it seems like these planes are carving up space into contiguous regions, sort of like Voronoi cells))

```python
nbits = 4  # Number of hyperplanes and binary vals to produce
d = 2  # vector dimensions

import numpy as np
# create a set of 4 hyperplanes, with 2 dimensions
plane_norms = np.random.rand(nbits, d) - .5

```

![[Pasted image 20240613141537.png]]

### Hashing Vectors
Now let's add three vectors and work through building our hash values using our four normal vectors and their hyperplanes.

```python
a = np.asarray([1,2])
b = np.asarray([2,1])
c = np.asarray([3,1])

# Calculate the dot product for each of these
a_dot = np.dot(a, plane_norms.T)
b_dot = np.dot(b, plane_norms.T)
c_dot = np.dot(c, plane_norms.T)
a_dot
# array([ 0.41487151, -0.32851916, -0.39600301, -0.16017455])  ... The ones that are >0 are True.

```

![[Pasted image 20240613142030.png]]
This produces our hashed vectors. Now, LSH uses these values to create buckets (which will contain some reference back to our vectors). Note that we don't store the original vectors in the buckets, which would significantly increase the size of our LSH index.
- In implementations like FAISS, the position/order that we added the vector is usually stored.

```python
vectors = [a_dot, b_dot, c_dot]
buckets = {}
i = 0

for i in range(len(vectors)):
    # convert from array to string
    hash_str = ''.join(vectors[i].astype(str))
    # create bucket if it doesn't exist
    if hash_str not in buckets.keys():
        buckets[hash_str] = []
    # add vector position to bucket
    buckets[hash_str].append(i)

print(buckets)
# {'1000': [0], '0110': [1, 2]}

```
Let's say we introduce a query vector that gets hashed as `0111`
- With it, we compare to every bucket in our LSH index, which in this case is only `1000` and `0110`.  We then use [[Hamming Distance]] to find the closest match, which happens to be `0110`.
- ![[Pasted image 20240613142504.png|300]]
- We've taken a linear complexity function which required us to compute the distance between our query vector and all previously indexed vectors, to a sub-linear complexity (we only compare to the grouped bucket hashes).
	- At the same time, vectors 1 and 2 are both equal to 0110, so we can't find which of these is closest to our query vector, so there's a degree of search quality being lost. We trade quality for speed.
	- ((Note that in this case there's no fully-dimensioned dense vector to use on the subset of bucketed vectors. The only dense vectors we have are these ones created using random hyperplane projections.))
- We control resolution using the `nbits` value; higher values increases the resolution of hashed vectors.
	- ![[Pasted image 20240613142815.png|300]]
Adding more possible combinations of hash values increases the potential number of buckets, increasing the number of comparisons, and, therefore, search time.
It's worth noting that not all bucket will generally be used, especially with higher `nbits` values.
- We see through our FAISS implementation that an `nbits` value of `128` or more is completely valid and still faster than using a  flat index.

### LSH in FAISS
- We use the `IndexLSH` index
Assume we have some wb dataset
```python
import faiss

d = wb.shape[1]
nbits = 4

# Initialize the index using our vectors dimensionality (128) and nbits
index = faiss.IndexLSH(d, nbits)
# then add the data
index.add(wb)

# Now we can search

D, I = index.search(xq, k=10)
# I = Index positions (row numbesr from wb) of our best k
# D = distances between thsoe best matches and our xq query
```

### Where to use LSH
- LSH can be a swift index, but it's less accurate than a Flat index. Using the Sift1M dataset, their best recalls core was achieved using an `nbits` value of 768 (even better recall is possible at excessive search times)
![[Pasted image 20240613143434.png|200]]


## Chapter 5: Product Quantization
- Quantization is a generic method referring to the compression of data into a smaller space. It's different from dimensionality reduction, which tries to represent data using *fewer* numbers; instead, quantization targets the scope/granularity of values.
	- There are many ways of doing this -- for example, we can Cluster a set of of vectors, replacing the larger scope of potential values with a smaller, discrete, and symbolic set of centroids.
- [[Product Quantization]] (PQ) is used to reduce the memory footprint of indices (it's not the only method to do this, but other methods don't manage to reduce memory size as effectively as PQ.)
	- A ==code== refers to a quantized representation of our vectors.
- PQ is the process of:
	1. Taking a big, high-dimensional vector
	2. Splitting it into equally-sized chunks (subvectors)
	3. Assign each subvector to its nearest centroid (also called reproduction/reconstruction values)
	4. Replacing these centroid values with unique IDs, each representing a centroid ((And then I think re-concatting these IDs into a dense vector))
[Animation Link](https://d33wubrfki0l68.cloudfront.net/af00b6764682bea50979e2285c0380f99e06466e/48940/images/product-quantization-6.mp4)

At the end of the process, we've reduced our high-dimensionality vector into a tiny vector of IDs requiring very little memory.

Usually when we're clustering, we optimize our k cluster centroids to split our vectors into (eg) 3 categories based on each vector's nearest centroid. For PQ, we do the same thing with one minor difference -- ==each subvector space (subspace) is assigned its own set of clusters; so we produce a set of clustering algorithms across multiple subspaces.==

Each of our subvectors are assigned to one of these centroids. In PQ terminology, these centroids are called reproduction values, represented by $(c_j, i)$, where $j$ is our subvector identifier, and $i$ identifies the chosen centroid (where there are $k*$ centroids for each subvector space $j$).

![[Pasted image 20240613150813.png|300]]
When we process a vector with PQ, it's split into subvectors, those subvectors are then processed according to their nearest (sub)cluster centroids. Rather than storing our new quantized vector as being represented by the D*-dimensional centroids, we replace them with centroid IDs. These can later be mapped back to the full centroids by a `codebook`.

PQ implementation in FAISS is pretty straightforward, and we'll look at combining PQ with an inverted file (IVF) step to improve search speed.

```python
import faiss

D = xb.shape[1]  # original dim of vectors
m = 8 # Number of subvectors we'd like to split into
assert D % m == 0

# number of bits per subquantizer, k* = 2**nbits
nbits = 8  # AKA # centroids assigned to each subspace

index = faiss.IndexPQ(D, m, nbits)
index.is_trained # False
index.train(xb)
index.add(xb)  # (? This was missing, I added it)
dist, I = index.search(xq, k)
```
Lower recall rates are a major drawback of PQ, counteracted somewhat by using larger `nbits` values at the cost of slower search times and *very* slow index construction times.
- Higher recall is out of reach for both PQ and IVFPQ indices -- if higher recall is required, another index should be considered.

### IndexIVFPQ
- To speed up our search time, we can add another step, using an IVF index, which will act as the initial broad stroke in reducing the scope of vectors in our search. After this, we continue PQ as we did before, but with a significantly-reduced number of vectors; thanks to minimizing our search scope, we should find we get get vastly improved search speeds!
	- ((From the code, it seems to be that the IVFPQ are each/both serving as the outer wrappers to a Flat index?

```python
vecs = faiss.IndexFlatL2(D)
nlist = 2048  # number of voronoi cells (must be >= k*, which is 2**nbits)
nbits = 8 # When using IVF+PQ, higher nbits values are n't supported

index = IndexIVFPQ(vecs, D, nlist, m, bits)
```
- `nlist` above defines how many Voronoi cells we use to cluster our *already quantized PQ vectors!*
	- These Voronoi cells are simply a set of partitions! Similar vectors are assigned to different partitions (or cells); when it comes to search, we introduce our query vector xq and restrict our search to the nearest cell(s).
![[Pasted image 20240613152942.png|300]]
Above: ((So we've used PQ to reduce the dimensionality of our vectors, and then used IVF to basically cluster those vectors. Once we select a Voronoi cell(s), we then use a flat index)

We can increase our (initially poor) recall by using the `nprobe` parameter, which tells us *how many* of the nearest Voronoi cells to include in our search scope!
- We want to set the lowest `nprobe` value that achieves satisfactory recall performance.

## Chapter 6: Hierarchical Navigable Small Worlds (HNSW
- HNSW is a hugely popular technology that time and time again produces SoTA performance with super fast search speeds and fantastic recall.
- We can split [[Approximate Nearest Neighbor Search|ANN]] algorithms into three distinct categories:
	- Trees
	- Hashes
	- Graphs (this is where HNSW is; specifically, a *==Proximity Graph==*, where links are based on proximity; closer vertices are linked)
- Two fundamental techniques contributed heavily to HNSW making the jump from a Proximity Graph to a *hierarchical navigable small world* graph:
	1. Probability Skip List
	2. Navigable Small Worlds Graphs

### (1/2) Probability Skip List
- The probability skip list was introduced way back in 1990; it allows fast search like a sorted array, while using a linked list structure for easy/fast insertion of new elements (which is not possible with sorted arrays).
	- they work by building several layers of linked lists; as we move down layers, the number of "skips" by each link is decreased.
	- To search a skip list, we start at the highest layer with the longest "skips" and move along the edges toward the right; If we find that we've overshot, we move down to the previous node in the next level.
![[Pasted image 20240613153817.png|300]]
HNSW inherits this same layered format, with longer edges in the highest layers (for fast search) and shorter edges in the lower layers (for accurate search).

### (2/2) Navigable Small Worlds Graphs (NSW Graphs)
- Introduced in 2011-2014; the idea is that we take a proximity graph, but build it so that we have both long-range and short-range links, then search times are reduced to (poly/)logarithmic complexity.
- Each vertex in the graph connects to several other vertices - we call these connected vertices *friends*, and each vertex keeps a *friend list*.
- When searching an NSW graph, we have a pre-defined ==entry point==, which connects to several nearby vertices. We identify which of these is the closest to our query vector and move there. We repeat this greedy-routing search of moving from vertex to vertex by identifying the nearest neighboring vertices in each friend list. Eventually, we find no nearer vertices than our current vertex, which is our local minimum/stopping condition.
- ![[Pasted image 20240613154814.png|300]]

NSW small world models are defined as any network with logarithmic complexity using greedy routing; the efficiency breaks down for larger networks (1-10k+ vertices) when graphs aren't routable.

The routing consists on two phases:
1. "Zoom-out" phase where we pass through low-degree vertices
2. Later "Zoom-in" phase where we pass through higher-degree vertices.

![[Pasted image 20240613155109.png]]
Our stopping condition is when we find no nearer vertices in our current vertex's friend list. Because of this, we're more likely to hit local minimum and stop too early when in the "zoom-out" phase.

To minimize the probability of erroneously stopping early (and increase recall), we can increase the average degree of vertices (but this increases search time). We can also start the search on high-degree vertices.

### Creating HNSW
HNSW is a natural evolution of NSW, inspired by hierarchical multi-layers of the Skip-List struture.
- Adding hierarch to NSW produces a graph where links are separated across different layers. 
	- ==At the top layer, we have the longest links==
	- ==At the bottom layer, we have the shortest links.==

![[Pasted image 20240613155328.png|300]]
We traverse edges in each layer like we did for NSW, greedily moving to the nearest vertex until we find the local minimum.
- At this point, we shift to the current vertex in a lower layer, and begin searching again. ((Notice that all nodes present in layer `n` are present in layer `n-1`, so you can move "down" on any node.))
- We repeat this process until we find the local minimum of our bottom layer - layer 0.
![[Pasted image 20240613155935.png|400]]

### Graph Construction
- How do we construct this graph? Vectors are inserted one-by-one, and the number of layers is represented by a parameter L.
- The probability of vector insertion at a given layer is given by a probability function normalized by the "level multiplier"!
![[Pasted image 20240613160035.png|300]]
- The creators of HNSW found the best performance is achieved when we minimize the overlap of shared neighbors across layers. An optimal rule of thumb is `1/ln(M)`

Graph construction starts at the top layer. (Omitting some notes here)
![[Pasted image 20240613160230.png]]

### Implementation in FAISS
```python
d = 128 # vector size
M = 32

# By default, M_max and M_max0 are set to M, M*2 at index initialization
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.max_level  # -1 ; needs to be set
levels = faiss.vector_to_array(index.hnsw.levels)
np.bincount(levels)  # array([], ...) Layers empty too

# We ned to build the index and set those params
index.add(xb)
index.hnsw.max_level  # 4; level has been set automatically
np.bincount(faiss.vector_to_array(index.hnsw.levels))  # array([     0, 968746,  30276,    951,     26,      1], dtype=int64)
# levels shows the distribution of vertices on each level from 0 to 4 (ignoring the first 0 value).

# We can even see which vector is our entry point:
index.hnsw.entry_point # 118295

```

HNSW isn't the best index in terms of memory utilization, but we can improve it by compressing our vectors using product quantization (PQ) (Which will reduce recall and increase search times)

If instead we wanted to improve our search speeds, we can do that too, adding an IVF component to our index.


## Chapter 7: Composite Indices and the FAISS Index Factory

- In the world of vector search, there are many indexing methods and vector processing techniques allowing us to prioritize between recall, latency, and memory usage.
	- Specific methods like [[Inverted File Index|IVF]], [[Product Quantization|PQ]], or [[Hierarchical Navigable Small Worlds|HNSW]] often return good results, but for best performance, we might want to use composite indices.
	- For instance, we can use an inverted file index (IVF) to reduce the scope of our search (increasing search speed), and then add a compression technique like product quantization (PQ) to keep larger indices within a reasonable size limit.

Composite indices are built from any combination of:
- ==Vector transform==: A pre-processing step to apply to vectors before indexing (PCA, OPQ)
- ==Coarse quantizer==: A rough organization of vectors to sub-domains (for restricting search scope; IVF, IMI, HNSW). Refers to clustering vectors to enable non-exhaustive search by limiting search scope.
- ==Fine quantizer==: A finer compression of vectors into smaller domains (for compressing index size; PQ). Describes the compression of vectors into codes, reducing memory usage of the index.
- ==Refinement==: A final step at search-time which re-orders results using distance calculations on the original flat vectors. Alternatively, another index (non-flat) can be used.

![[Pasted image 20240613162536.png|300]]

It's often cleanest to build these composite indices using the FAISS `index_factory` class.


```python
# this...
quantizer = faiss.IndexFlatL2(128)
index = faiss.IndexIVFFlat(quantizer, 128, 256)

# can become THIS, using index_factory
index_f = faiss.index_factory(128, "IVF256,Flat")
# Above: We don't have to specify L2 distance because the index_factory uses L2 by default.
```

Why use Index Factory?
- Can depend on personal preference; if you prefer class-based index-building approaches, stick with it.
- But Index Factory can greatly improve the elegance and clarity of the code. Five lines becomes 1.

```python
d = xb.shape[1]
m = 32
nbits = 8
nlist = 256

# Initialize our OPQ and coarse+fine quantizer steps separately
opq = faiss.OPQMatrix(d, m)
# d now refers to the shape of the rotated vectors from OPQ (whic hare equal)
vecs = faiss.IndexFlatL2(d)
sub_index = faiss.IndexIVFPQ(vecs, d, nlist, m, nbits)
# Nowe we merge the preprocessing, coarse, and fine quantization steps
index = faiss.IndexPreTransform(opq, sub_index)
# Will will add all of the previous steps to our final refinement step
index = faiss.IndexRefineFlat(q)
# Train the index and the index vectors
index.train(xb)
index.add(xb)

# Above: Pretty complicated, right? We can rewrite it all using our index factory to get much simpler code!
d = xb.shape[1]
# in index factory, m=32, nlist=256, nbits=8 by default
index = faiss.index_factory(d, "OPQ32,IVF256,PQ32,RFLat")
# train and index vectors
index.train(xb)
index.add(xb)
```

IVFADC (name isn't explained, seems just like an IVF+PQ combo)
- ADC = [[Asymmetric Distance Computation]] (ADC); referred to as asymmetric because we compare our non-compressed xq query vector against previously indexed, compressed PQ vectors.
![[Pasted image 20240613164323.png]]
![[Pasted image 20240613164502.png]]

To implement it using index_factory:
```python
index = faiss.index_factory(d, "IVF256,PQ32x8")
index.train(xb)
index.add(xb)
D, I = index.search(xq, k)
recall(I)  # 30

# We can also increase index.nprobe to search more IVF cells, improving recall but slowing the search
index.nprobe = 8
D, I = index.search(xq, k)
recall(I)
```
With this, we create an IVFADC index with 256 IVF cells; each vector is compressed with PQ using m and nbits values of 32 and 8, respectively. PQ uses nbits == 8 by default so we can also write "IVF256,PQ32".
- _m: number of subvectors that original vectors are split into_
- _nbits: number of bits used by each subquantizer, we can calculate the number of centroids used by each subquantizer as_ _2**nbits_

### Optimized Product Quantization
- IVFADC and other indexes using PQ can benefit from [[Optimized Product Quantization]] (OPQ)
- OPQ works by rotating vectors to flatten the distribution of values across the subvectors used in PQ; this is particularly useful for unbalanced vectors with uneven data distributions.
- In FAISS, we add OPQ as a pre-processing step:
	- In IVFADC, the OPQ index string looked like: `"OPQ32,IVF256,PQ32"` where hte 32 in OPQ32 and PQ32 refers to the number of bytes `m` in the PQ-generated codes.
	- _The OPQ matrix in Faiss is_ **_not_** _the whole rotation and PQ process. It is only the rotation. A PQ step must be included downstream for OPQ to be implemented._
```python
# we can add pre-processing vector rotation to
# improve distribution for the PQ step using OPQ
index = faiss.index_factory(d, "OPQ32,IVF256,PQ32x8")
index.train(xb)
index.add(xb)
D, I = index.search(xq, k)
recall(I)  # 31, a 1-point gain! In our case, hte data distribution of the Sift1M dataset is already well-balanced, so OPQ only gives us a minor 1% increase in recall.
```
If we wanted to increase the nprobe value to improve recall, we can no only access nprobe directly with index.nprobe, since our index value no longer refers to the IVF portion of our index (since we added a pre-processing step to our index).
- Instead, we have to extract the IVF index before modifying the nprobe value:
```python
ivf = faiss.extract_index_ivf(index)
ivf.nprobe = 13
D, I = index.search(xq, k)
recall(I)  # 74; woo (Search time increased from 720us to 1060us).
```

### Multi-D-ADC
- Refers to Multi-dimensional indexing, alongside a PQ step which produces an Asymmetric Distance Computation at search time.
- Based on the inverted multi-index (IMI), an extension of IVF.
	- IMI can outperform IVF in both recall and search speed, but *does* increase memory usage.
	- Makes IMI indices (like multi-D-ADC) ideal in cases where IVFADC doesn't quite reach the speed and recall required, and you can spare more memory usage.
	- IMI works very similarly to IVF, but Voronoi cells are split across vector dimensions -- produces something equivalent to multi-level vornoi cell structures.
![[Pasted image 20240613170146.png|300]]

When we add a vector compression to IMI using PQ, we produce the multi-D-ADC index; where ADC refers to the asymmetric distance computation that is made when comparing query vectors to PQ vectors.

```python
index = faiss.index_factory(d, "IMI2x8, PQ32")
index.train(xb)
index.add(xb)

imi = faiss.extract_index_ivf(index)  # access nprobe
imi.nprobe = 620
D, I = index.search(xq, k)
recall(I)
```

### HNSW Indexes
- IVF with HNSW is our final composite index; it splits our indexed vectors into cells as usual per IVF, but this time the process is organized using HNSW.
- Produces comparable or better speed and *significantly higher recall* than our previous two indices, but at the cost of *much* higher memory usage.
![[Pasted image 20240613170500.png]]
- Using vanilla IVF, we introduce our query vector and compare it to every cell centroid, identifying the nearest centroid(s) to restrict our search scope with.
	- Pairing this with HNSW, we produce an HNSW graph of all of these cell centroids, making the exhaustive centroid search *approximate* (so the search for which centroid is approximated; this makes sense mostly with large numbers of clusters, where exhaustively searching cluster centroids with IVF would be too expensive)

With IVF+HNSW, we swap "few centroids and large cells" for "many centroids and small cells"
- ==We quickly approximate the nearest cell centroids using HNSW, the restrict our exhaustive search to those nearest cells.==
The standard IVF+HNSW index can be built with `"IVF4096_HNSW32,Flat"`.

```python
index = faiss.index_factory(d, "IVF4096_HNSW32,Flat")
index.train(xb)
index.add(xb)
D, I = index.search(xq, k)
recall(I)  # 25; let's increaes nprobe

index.nprobe = 146  # We can directly access
D, I = index.search(xq, k)
recall(I)  # 100; nice!
```
With this index, we can produce incredible performance ranging from 25% -> 100% recall at search times of 58.9µs -> 916µs.

However, the IVF+HNSW index is not without its flaws. Although we have incredible recall and fast search speeds, the memory usage of this index is ==_huge_==. Our 1M 128-dimensional vectors produce an index size of 523MB+.
- We can reduce this using PQ and OPQ, but this will reduce recall and increase search times.