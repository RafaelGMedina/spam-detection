# Dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder
# from tensorflow.models import Sequential 


# Getting Data Ready
path = "sms+spam+collection/SMSSpamCollection"

df = pd.read_csv(path, sep=r"\t", header=None, engine="python")
df = df.rename({0:"Class", 1:"SMS"}, axis=1)

# Fn to clean sms in terms of spacings

def clean_str(txt):
    txt = txt.lower()

    modified_str = ""
    for i in range(len(txt)):
        # Make sure there is only one space between words
        if (i == 0) and (txt[i] != " "):
            modified_str += txt[i]
        elif (i == len(txt)-1) and (txt[i] != " "):
            modified_str += txt[i]
        elif (txt[i] == " ") and (txt[i+1] != " ") and (i != 0):
            modified_str += " "
        elif txt[i] != " ":
            modified_str += txt[i]

    # return with removed punctuation
    return modified_str.translate(str.maketrans("", "", string.punctuation))


df["SMS"] = df["SMS"].apply(clean_str)

preprocessed_sentences = [df['SMS'][i].split(' ') for i in range(df.shape[0])]


# Theory: Message format could also be a signal for spam/ham; poor format could indicate spam
# The use of word shorteners could indicate ham


# Convert Spam/Ham to 1/0

df["Class"] = df["Class"].replace({"spam":1, "ham":0})

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
'''
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