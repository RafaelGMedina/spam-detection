import pandas as pd
import numpy as np
from classes import Huffman

test_data = pd.DataFrame({'Word': ['a', 'b', 'c', 'd', 'e', 'f'], 'Count': [5, 9, 12, 13, 16, 45]})


'''
# First, create Huffman objects for each of the words and replace the Words column with the objects instead of the word

huff_objs = pd.Series([Huffman(5, word) for word in test_data['Words']])

test_data['Words'] = huff_objs

print(test_data)

'''
huff = Huffman(test_data)

huff.fit()

print(huff.root)

print(huff.code)