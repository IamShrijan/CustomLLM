import os
import glob
import re

# Define the path to the DataSet folder
folder_path = 'DataSet'

# Use glob to find all text files in the folder
text_files = glob.glob(os.path.join(folder_path, '*.txt'))

# Initialize a variable to store the contents
all_text_contents = ""

# Iterate over each file and read its contents
for file_path in text_files:
    with open(file_path, 'rb') as file:  # Open in binary mode
        content = file.read()
        all_text_contents += content.decode('utf-8', errors='ignore') + "<|endoftext|>"
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', all_text_contents)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[-1])

all_words = sorted(set(preprocessed))
all_words.extend(["<|unk|>"])
vocab_size = len(all_words)

print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
tokenizer = SimpleTokenizer(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)
token = tokenizer.encode(text)
print(token)
text = tokenizer.decode(token)
print(text)
