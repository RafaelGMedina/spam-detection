import pandas as pd

class Node:
    def __init__(self, value=None, count=None, left=None, right=None):
        self.value = value
        self.count = count
        self.left = left
        self.right = right
        

# Huffman Tree
class Huffman:
    def __init__(self, data: pd.DataFrame):
        # data should be a DataFrame with two columns. The first is a column of node objects and the second is count
        self.data = data
        self.root = None
        self.code = None

    
    def encoding(self, tree: Node, code=''):

        if tree.value != None:
            return {tree.value: code}
        
        left_traverse = self.encoding(tree.left, code=code+'0')
        right_traverse = self.encoding(tree.right, code=code+'1')

        return left_traverse | right_traverse
        
    def fit(self):
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
            new_node = Node(count=current_count, left=first_min, right=second_min)

            # Insert the new node and count into our dataframe
            new_row = pd.Series({'Word': new_node, 'Count': current_count})
            self.data = pd.concat([self.data, new_row.to_frame().T], ignore_index=True)

        self.root = self.data['Word'][0]

        # Create the encoding for the tree using the 'encoding' function

        self.code = self.encoding(self.root)