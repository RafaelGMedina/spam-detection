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
vocab = list(vocab_for_huffman.value_counts().index)


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
    pairs = {}

    for word in sentence:
        
        pairs[word] = []

    for word_idx in range(len(sentence)):
        current_target = sentence[word_idx]
        count = 1

        while count != window_size+1:
            left = word_idx - count
            right = word_idx + count
            if left >= 0:
                pairs[current_target].append(sentence[left])
            if right < len(sentence):
                pairs[current_target].append(sentence[right])

            count += 1

    return pairs



df['Training Pairs'] = df['SMS'].apply(lambda x: context_target(x, 2))

# Our DataFrame looks like [Class, SMS (Each SMS is a list, each element is a word), Training Pairs]
print(df)
# Construct Huffman Tree
'''
huff = Huffman(vocab_for_huffman)
huff.fit()
'''




























'''


# Use gensim's Word2Vec model

model = Word2Vec(sentences=preprocessed_sentences, vector_size=50, window=5, min_count=1, workers=4, sg=1)

print(model.wv.similarity('free', 'money'))

X = df["SMS"]
y = df["Class"]

"""
# Word Embedding from Scratch

# Get the vocabulary we are training our model on
ohe = OneHotEncoder()
vocab = list(set(X.str.split(" ").sum()))
# Lowercase every character

for i in range(len(vocab)):
    vocab[i] = vocab[i].lower()

vocab = np.array(vocab).reshape(len(vocab), 1)
ohe.fit(vocab)



"""


# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.85, random_state=1)


# Create a NN that can take a word with context and guess the "sentiment" of the phrase. 

# Maybe start with 4 words (input size 4)

# First task, tokenize the SMS' so that each record has some number of sets of 4 words each

# Function to tokenize

def tkn(string):
    string = string.split(" ")
    if len(string)%4 != 0:
        index = -len(string)%4
        remainder = string[-4:]

        string = string[:index]

        # New String
        string = string+remainder
    groups = len(string)/4
    tknize = [string[(x-1)*4:x*4] for x in np.arange(1, groups+1)]

    return tknize


# Now, build a RNN (simple)
# 4-to-1

# NOTE: I already have embeddings for all the words that appear in the 
# all the SMS'. What I'm going to do now is use the groups of 4 words to train 
# a simple RNN. 

# First: Split up the SMS' to groups of 4

grouped_X_train = X_train.apply(tkn)
grouped_X_test = X_test.apply(tkn)

# Convert to List

group_list_train = [grouped_X_train.iloc[i] for i in range(X_train.shape[0])]
group_list_test  = [grouped_X_test.iloc[i] for i in range(X_test.shape[0])]

# Convert the words into their respective embeddings

for sms in range(len(group_list_train)):
    for group in range(len(group_list_train[sms])):
        for word in range(len(group_list_train[sms][group])):
            group_list_train[sms][group][word] = w2v_model.wv[group_list_train[sms][group][word]]


# Each training "sample" will be one group of 4 words for the RNN

# Create an nx4x300 matrix, where each row is a group of 4 words, each represented by their embedding. 

# List ordering
# 1st: SMS
# 2nd: Group
# 3rd: Word embedding

embedded_df = pd.DataFrame({"SMS": group_list_train, "Class": y_train})

# train the RNN



'''