import tiktoken
import os
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

tokenizer = tiktoken.get_encoding("gpt2")

folder_path = 'DataSet'

# Use glob to find all text files in the folder
text_files = glob.glob(os.path.join(folder_path, '*.txt'))

# Initialize a variable to store the contents
raw_text = ""

# Iterate over each file and read its contents
for file_path in text_files:
    with open(file_path, 'rb') as file:  # Open in binary mode
        content = file.read()
        raw_text += content.decode('utf-8', errors='ignore') + "<|endoftext|>"
#preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
#preprocessed = [item.strip() for item in preprocessed if item.strip()]
#print(preprocessed[-1])

#enc_text = tokenizer.encode(raw_text,allowed_special={'<|endoftext'}, disallowed_special=())
#print(len(enc_text))

# Initialize the encodings for GPT-2, GPT-3, and GPT-4
#available encodings.
encodings = {
    "gpt2": tiktoken.get_encoding("gpt2"),
    "gpt3": tiktoken.get_encoding("p50k_base"),  # Commonly associated with GPT-3 models
    "gpt4": tiktoken.get_encoding("cl100k_base")  # Used for GPT-4 and later versions
}

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"},  disallowed_special=())

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length] 
            target_chunk = token_ids[i + 1: i + max_length + 1] # for every input we are predicting the next word
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

vocab_size = 50257 # Vocab size of gpt2 tokenizer
sentence_size = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=sentence_size,
    stride=4, shuffle=False
)

output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(sentence_size, output_dim)


pos_embeddings = pos_embedding_layer(torch.arange(sentence_size))
