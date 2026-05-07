import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec



# Sample sentences with variety
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world we live in.",
    "Deep learning techniques have greatly improved image recognition.",
    "Natural language processing allows computers to understand human language.",
    "Data science combines statistics, computer science, and domain knowledge.",
    "The weather is nice today, perfect for a walk in the park.",
    "Cats are often seen as independent and curious creatures.",
    "The stock market fluctuates based on various economic indicators.",
    "Exploring new cuisines can be an exciting culinary adventure.",
    "Machine learning algorithms can learn from data and make predictions.",
]
# Preprocess sentences
test = []

for sentence in sentences:
    test.append(sentence.split(" "))

# Train Word2Vec model
model = Word2Vec(sentences=test, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Save the model for future use
print(model.wv.index_to_key)