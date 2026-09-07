import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from math import ceil

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
            self.vec = None
        

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
        # Data looks like: [Class, SMS (list of words), Training Pairs]
        self.data = data
        self.huffman = huffman
        self.vocab = vocab
        self.ohe = OneHotEncoder().fit(np.array(self.vocab).reshape(-1, 1))
        self.training_pairs = []

        # Put all of the training pairs into a single list
        for row in range(self.data.shape[0]):
            self.training_pairs += self.data['Training Pairs'].iloc[row]

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    # The update equation that will be used to update the nodes of the huffman tree
    def update_equation(old_vec, learning_rate, derivative):
        return old_vec - learning_rate*derivative

    # The algorithm will be used when updating the huffman vector for a single node
    # The logic for updating all nodes will be in a different function
    def dE_dv(self, v: np.array, h: np.array, direction):
        return self.sigmoid(v.T@h) - direction

    def node_update(self, tree: Huffman, training_paths, H, batch_size):
        # We might use this function to recursively traverse the tree and calculate updates
        #
        # Logic: All words will start at the root, so when calculating the update, we take all the words into consideration. 
        # Then some will diverge left, the rest right. Once this divergence occurs, we will call the function for the 
        # nodes on the left path and another call for the nodes on the right path. The base case for any path is if we have reached 
        # the word we are looking for. We only call the function on words whose path still contains paths. When making the function
        # calls, we will need to 'pop' the previous decision
        #
        # To preserve the original weights, we will need to perform this updating after we update the input to hidden weights. 

        H = tree.vec

        # Base Case: H/training_paths is empty. We don't return anything since we are updating the nodes on the fly 
        if H == None:
            return

        # We will update the node here


        # Second Part
        left_group_H = []
        left_group_train = []
        right_group_H = []
        right_group_train = []
        

        for sample_idx in range(len(H)):
            direction = training_paths[sample_idx][0]

            if direction == 1:
                left_group_H.append(H[sample_idx])
                left_group_train.append(training_paths[sample_idx])
            elif direction == 0:
                right_group_H.append(H)
                right_group_train.append(training_paths[sample_idx])

            if len(training_paths[sample_idx]) == 1:
                training_paths.pop(sample_idx)
                H.pop(sample_idx)

            else:
                H[sample_idx] = H[sample_idx][1:]



        pass



    # This will be the main function that will be used for training our word embeddings using the above helper functions
    def main(self, batch_size, embedding_dim):
        # Put all training pairs into a single list
        num_samples = len(self.training_pairs)


        # 1: Set up the batch samples
        batch_samples = []
        start_idx = 0
        groups = ceil(num_samples/batch_size)

        for batch in range(groups):
            if batch == groups-1:
                batch_samples.append(self.training_pairs[start_idx:])
            else:
                batch_samples.append(self.training_pairs[start_idx: start_idx+batch_size])
                start_idx += batch_size

        # 2: Instantiate the word embeddings/weight matrix

        # Vocab x Embed matrix
        W = np.array([[.5]*embedding_dim for i in range(len(self.vocab))])
        # For the input matrix, since it's a OHE, and H will simply be a subset of W. We only need the column of W


        # This is where the training will occur. Weight update will be after every batch

        # TODO: Will need to optimize this so that we don't have a double for loop
        for batch in batch_samples:
            words = []
            targets = []
            huffman_path = []

            # unpack the dictionaries into two lists
            for tup in batch:
                target = tup[1]

                words.append(tup[0])
                targets.append(target)

                huffman_path.append(self.huffman.code[target])

            to_transform = np.array(words).reshape(-1, 1)
            input_columns: np.array = self.ohe.transform(to_transform).indices
            
            H = W[input_columns]
            # We will only update the weights once we iterate through all the samples in the batch

            for i in range(len(words)):
                dE_dv = 0

                current_word = words[i]
                current_targets = targets[i]

                for target_word in current_targets:
                    path_to_target = self.huffman.code[target_word]




