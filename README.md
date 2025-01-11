# SBERT-Sentence-Transformers
SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings.
Research Paper Link:<ins>https://arxiv.org/pdf/1908.10084.pdf </ins>

- It is superfast 5 seconds vs 50 hours with BERT
- It can be used for both classification and regression

## 1. Semantic Search
This project demonstrates semantic search using SBERT embeddings. It fetches a corpus and queries from two text files, encodes them using SBERT, normalizes the embeddings, and performs a semantic search to find the most relevant corpus sentences for each query based on cosine similarity.
### Installation 
To set up the environment, install the required library:  
```bash
pip install -U sentence-transformers
```
### How It Works
Model: Uses the `multi-qa-MiniLM-L6-cos-v1` model for semantic search
### Processes
- Fetch the corpus and queries from URLs.
- Encode the corpus and queries into embeddings using SBERT.
- Normalize the embeddings for efficient similarity calculation.
- Perform semantic search to find the top 3 relevant corpus sentences for each query.
- Display the results with similarity scores.
### Output
```
Query: "A man is eating pasta."
Result: 
- A man is eating food. --> 0.851
- A man is eating a piece of bread. --> 0.841
- A man is eating pasta. --> 0.856
```
## 2. Sentence Embeddings and Similarities
This project demonstrates how to compute sentence embeddings using the `sentence-transformers` library and measure semantic similarities between sentences. Additionally, it identifies paraphrases based on similarity scores.  
### Installation  

To set up the environment, install the required library:  
```bash
pip install -U sentence-transformers
```
### How It Works
Model: Uses the `all-MiniLM-L6-v2` model for sentence embeddings.

### Input Sentences
```
sentences = ['the cat sits outside', 'the new movie is awesome', 'the new movie is really great', 'the dog bark on strangers']  
```

### Processes
- Generates embeddings for each sentence.
- Computes cosine similarity scores for sentence pairs.
- Identifies paraphrases with high similarity.

### Output
Cosine Similarity Matrix
```
Example output for the similarity matrix:
[[1.00, 0.23, 0.22, 0.15],  
 [0.23, 1.00, 0.89, 0.17],  
 [0.22, 0.89, 1.00, 0.18],  
 [0.15, 0.17, 0.18, 1.00]]  
```

Example paraphrases with similarity scores:
```
'the new movie is awesome' <> 'the new movie is really great' --> 0.89
```
## 3. Clustering
Sentence Embeddings are used for clustering tasks, for:
- K-Means
- Agglomerative Clustering
- Fast Clustering using Sentence Transformers.
### Installation
```bash
!pip install sentence_transformers
!pip install requests
!pip install scikit-learn
!pip install pandas
```
### How It Works
1. Sentence Embedding
We use the SentenceTransformer model from the sentence-transformers library to convert sentences into embeddings. The embeddings represent the semantic meaning of each sentence, allowing for better clustering based on context.

2. Clustering Algorithms
   - K-Means Clustering: This divides the sentences into a predefined number of clusters by minimizing the variance within each cluster.
   - Agglomerative Clustering: This is a hierarchical clustering method that builds the hierarchy from bottom-up and can determine the number of clusters  
     dynamically based on a distance threshold.
   - Fast Clustering: This leverages a fast community detection algorithm to group similar questions together based on cosine similarity of their embeddings.
### Input Corpus
```
['A man is eating food.', 'A man is eating a piece of bread.', 'A man is eating pasta.', 'The girl is carrying a baby.', 'The baby is carried by the woman', 'A man is riding a horse.', 'A man is riding a white horse on an enclosed ground.', 'A monkey is playing drums.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah is running behind its prey.', 'A cheetah chases prey on across a field.', '']
```
### Input Clusterings
1. K-Means

   ```
   Cluster  1
   ['A monkey is playing drums.', 'Someone in a gorilla costume is playing a set of 	drums.']

   Cluster  2
   ['A man is eating food.', 'A man is eating a piece of bread.', 'A man is eating pasta.', 'A 
   man is riding a horse.', 'A man is riding a white horse on an enclosed ground.']
   ....
   ```


2. Agglomerative Clustering

   ```
   Cluster  1
   ['A man is eating food.', 'A man is eating a piece of bread.', 'A man is eating pasta.']

   Cluster  2
   ['A man is riding a horse.', 'A man is riding a white horse on an enclosed ground.']

   Cluster  3
   ['A monkey is playing drums.', 'Someone in a gorilla costume is playing a set of drums.']

   Cluster  4
   ['The girl is carrying a baby.', 'The baby is carried by the woman']
   .....
   ```


3. Fast Clustering

   ```
   Cluster 1, #10 Questions
	 Which are the best Hollywood thriller movies?
	 What are the most underrated and overrated movies you've seen?
	 What are the best films that take place in one room?
	 ...

   Cluster 2, #9 Questions
	 What are your views on Modi governments decision to demonetize 500 and 1000 rupee 
         notes? How will this affect economy?
	 What's your opinion about the decision on removal of 500 and 1000 rupees currency 
         notes?
	 How will Indian GDP be affected from banning 500 and 1000 rupees notes?
	 ...

   Cluster 3, #8 Questions
	 What is best way to make money online?
	 How can I make money through the Internet?
	 What are the easy ways to earn money online?
    .....
   ```
