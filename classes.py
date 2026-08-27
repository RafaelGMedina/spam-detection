import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Node:
    def __init__(self, value=None, count=None, left=None, right=None, embed_dim=None):
        self.value = value
        self.count = count
        self.left = left
        self.right = right
        # instantiate initial node vectors
        try:
            self.vec = [.5 for i in range(embed_dim)]
        except:
            pass
        

# Huffman Tree
class Huffman:
    def __init__(self, data: pd.Series, embed_dim):
        # The data is simply a Series of value counts of the words in the whole corpus
        self.data = data
        self.root = None
        self.code = None
        self.embed_dim = embed_dim


    
    def encoding(self, tree: Node, code=''):

        if tree.value != None:
            return {tree.value: code}
        
        left_traverse = self.encoding(tree.left, code=code+'0')
        right_traverse = self.encoding(tree.right, code=code+'1')

        return left_traverse | right_traverse
        
    def fit(self):
        # Part 0: Convert the count series into a usable dataframe
        words = list(self.data.reset_index().loc[:, 0])
        counts = list(self.data)
        self.data = pd.DataFrame({'Word': words, 'Count': counts})
        # Part 1: Create Node objects for each word
        new_col = []
        for row in range(self.data.shape[0]):
            current_val = self.data.iloc[row]['Word']
            new_col.append(Node(value=current_val, count=self.data.iloc[row]['Count']))
            
        self.data['Word'] = new_col

        # Part 2: Construct the tree
        while self.data.shape[0] != 1:
            self.data = self.data.sort_values(by='Count')

            first_min = self.data['Word'].iloc[0]
            self.data = self.data[1:]
            second_min = self.data['Word'].iloc[0]
            self.data = self.data[1:]

            current_count = first_min.count + second_min.count
            new_node = Node(count=current_count, left=first_min, right=second_min, embed_dim=self.embed_dim)

            # Insert the new node and count into our dataframe
            new_row = pd.Series({'Word': new_node, 'Count': current_count})
            self.data = pd.concat([self.data, new_row.to_frame().T], ignore_index=True)

        self.root = self.data['Word'][0]

        # Create the encoding for the tree using the 'encoding' function

        self.code = self.encoding(self.root)


# Class that trains a Skip-Grams NN
class EmbeddingNN:
    def __init__(self, data: pd.DataFrame, huffman: Huffman, vocab: list):
        # Data looks like: [Class, SMS (list of words), Training Pairs (dictionary of training pairs (input: [targets]))]
        self.data = data
        self.huffman = huffman
        self.ohe = OneHotEncoder().fit(vocab)

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    # The update equation that will be used to update the nodes of the huffman tree
    def update_equation(old_vec, learning_rate, sigmoid, direction, var):
        # Depending on whether we are updating wrt the weight matrix or the weights of the nodes, 
        # var will either be the hidden layer or the node vector
        return old_vec - learning_rate*(sigmoid - direction)*var

    # This will be the main function that will be used for training our word embeddings using the above helper functions
    def main(self, batch_size):
        # self.data contains the Training Pairs column that we will use for updating
        
        # TODO split the data into the desired batch sizes


        # test

        pass