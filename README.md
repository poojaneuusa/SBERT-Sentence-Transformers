# SBERT-Sentence-Transformers
SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings.
Research Paper Link:<ins>https://arxiv.org/pdf/1908.10084.pdf </ins>

- It is superfast 5 seconds vs 50 hours with BERT
- It can be used for both classification and regression

## 1. Semantic Search

## 2. Sentence Embeddings and Similarities
This project demonstrates how to compute sentence embeddings using the `sentence-transformers` library and measure semantic similarities between sentences. Additionally, it identifies paraphrases based on similarity scores.  
### Installation  

To set up the environment, install the required library:  
```bash
pip install -U sentence-transformers
```
### How It Works
Model: Uses the `all-MiniLM-L6-v2` model for sentence embeddings.

### Input Sentences:
```
sentences = ['the cat sits outside', 'the new movie is awesome', 'the new movie is really great', 'the dog bark on strangers']  
```

### Processes:
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
