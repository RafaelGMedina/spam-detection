# Dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from classes import Huffman

from typing import List


# Getting Data Ready
path = "sms+spam+collection/SMSSpamCollection"

df = pd.read_csv(path, sep=r"\t", header=None, engine="python")
df = df.rename({0:"Class", 1:"SMS"}, axis=1)
# Convert Spam/Ham to 1/0
df["Class"] = df["Class"].replace({"spam":1, "ham":0})

# Function to clean sms in terms of spacings

def clean_str(txt):
    txt = txt.lower()
    # remove leading and trailing whitespace
    txt = txt.strip()

    txt = re.sub(r'\s+', ' ', txt)
    return txt


df["SMS"] = df["SMS"].apply(clean_str)

# Create a list where each element is the word in the string
preprocessed_sentences = [df['SMS'][i].split(' ') for i in range(df.shape[0])]

# Theory: Message format could also be a signal for spam/ham; poor format could indicate spam
# The use of word shorteners could indicate ham

# EDA (Done in the Python Notebook)

# Word Embeddings from Scratch

# Get the vocabulary as a single list in order to be able to one hot encode
vocab = []

for sentence in preprocessed_sentences:
    vocab+=sentence
vocab_series = pd.DataFrame(vocab)
vocab_for_huffman = vocab_series.value_counts()
vocab = list(vocab_series[0])


df['SMS'] = pd.Series(preprocessed_sentences, name='SMS')

# Vector embedding parameters
d = 50
window = 4
ivec_size = len(vocab)
imatrix_dim = (ivec_size, d)

# Create a function that will create the context/target pairs depending on our window size

def context_target(sentence: List[str], window_size):
    # window_size refers to window_size word(s) on the left of the current word and window_size
    # word(s) on the right of the current word
    pairs = []

    for word_idx in range(len(sentence)):
        current_target = sentence[word_idx]
        count = 1


        while count != window_size+1:
            left = word_idx - count
            right = word_idx + count
            if left >= 0:
                pairs.append((current_target,sentence[left]))
            if right < len(sentence):
                pairs.append((current_target, sentence[right]))

            count += 1        

    return pairs



df['Training Pairs'] = df['SMS'].apply(lambda x: context_target(x, 2))

# Our DataFrame looks like [Class, SMS (Each SMS is a list, each element is a word), Training Pairs]
# Construct Huffman Tree
'''
huff = Huffman(vocab_for_huffman)
huff.fit()
'''
