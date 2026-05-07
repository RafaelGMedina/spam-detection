import numpy as np
import string
some_string = "this is a string that doesn't divide into 4 evenly."



def tkn(string):
    string = string.split(" ")
    if len(string)%4 != 0:
        rem = len(string)%4
        index = -rem
        remainder = string[-4:]

        string = string[:index]

        # New String
        string = string+remainder
    groups = len(string)/4
    tknize = [string[(x-1)*4:x*4] for x in np.arange(1, groups+1).astype(int)]

    return tknize


# list comp.


def str_2_int(_2d_list):
    conversion = []
    for lst in _2d_list:
        current_group = ["".join([str(ord(char)) for char in [*word]]) for word in lst]

        conversion.append(current_group)
    return conversion







txt = "This is a test?string."

#print(txt.translate(str.maketrans(string.punctuation, "                                ")))

txt = "test  tone "
for i in range(len(txt)):
    # Make sure there is only one space between words
    if (i == 0) and (txt[i] == " "):
        print("True")
    
    


