import pandas as pd
import numpy as np
from classes import Huffman
from main import context_target

test_data = pd.DataFrame({'Word': ['a', 'b', 'c', 'd', 'e', 'f'], 'Count': [5, 9, 12, 13, 16, 45]})


test = context_target(['this', 'is', 'a', 'test'], 2)

huff = Huffman()