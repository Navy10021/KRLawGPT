# pip install transformers
import os
import numpy as np
from transformers import PreTrainedTokenizerFast

# Parameters for preparing dataset
split_ratio = 0.9
using_LLMs = False
reduce_size = False

# Build dataset and vocab function
def prepare_data(my_data, split_ratio, using_LLMs, reduce_size):
    """
    split_ratio : train | val split ratio
    using_LLMs : if using GPT huggingface tokenizer or not
    reduce_size : reduce text dataset size or not 
    """
    # 1. Dataset Split : train : test = 90% : 10%
    split_i = int(split_ratio * len(my_data))   
    train_data, val_data = my_data[:split_i], my_data[split_i:]

    # 2. Encoding
    # 2-1. using GPT-based vocab
    if using_LLMs:
        model_type = 'skt/kogpt2-base-v2'     # KoGPT-2 from SKT-AI
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_type)
        if reduce_size:
            train_data = train_data[:1000000]
            val_data = val_data[:20000]
        train_ids = tokenizer.encode(train_data)
        val_ids = tokenizer.encode(val_data)
        vocab_size = 51200
    # 2-2. Building my legal vocab
    else:
        chars = sorted(list(set(my_data)))
        vocab_size = len(chars)
        # mapping chars -> ints
        char_to_int = { char : i for i, char in enumerate(chars) }
        int_to_char = { i : char for i, char in enumerate(chars) }
        def encode(s):
            return [char_to_int[c] for c in s] 
        def decode(l):
            return ''.join([int_to_char[i] for i in l])
        train_ids = encode(train_data) 
        val_ids = encode(val_data)

    print(">> KRLawGPT vocab size : {}".format(vocab_size)) 
    print(">> Train dataset has {} tokens".format(len(train_ids)))
    print(">> Val dataset has {} tokens".format(len(val_ids)))


    # 3. Export to bin files
    data_dir = "./data"
    train_ids = np.array(train_ids, dtype = np.uint16)
    val_ids = np.array(val_ids, dtype = np.uint16)
    train_ids.tofile(os.path.join(data_dir, 'train_law.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val_law.bin'))

    return train_ids, val_ids, vocab_size


# Load raw text Dataset
with open('./data/korLaw.txt', 'r', encoding = 'utf-8') as f:
    text_data = f.read()

# Make Legal dataset for training and validation    
train_ids, val_ids, my_vocab_size = prepare_data(text_data, split_ratio = split_ratio, using_LLMs = using_LLMs, reduce_size = reduce_size)

data_dir = "./data"
train_data = np.memmap(os.path.join(data_dir, 'train_law.bin'), dtype = np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val_law.bin'), dtype=np.uint16, mode='r')
print("\n>> Legal text dataset is ready !")