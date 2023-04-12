import torch
from transformers import PreTrainedTokenizerFast

# Load our KRLawGPT model
model = torch.load("./output" + "/KRLawGPT.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def legal_text_generator(input_text, num_samples = 10, using_LLMs = False, max_new_tokens = 500, temperature = 0.8, top_k = 200):
    
    # 1.Tokenizer & Encoder & Decoder
    if using_LLMs:
        model_type = 'skt/kogpt2-base-v2'
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_type)
        encode = lambda s: tokenizer.encode(s)
        decode = lambda l: tokenizer.decode(l)

    else:
        with open('./data/korLaw.txt', 'r', encoding = 'utf-8') as f:
            text_data = f.read()
        chars = sorted(list(set(text_data)))
        char_to_int = { char : i for i, char in enumerate(chars) }
        int_to_char = { i : char for i, char in enumerate(chars) }
        encode = lambda s: [char_to_int[char] for char in s]
        decode = lambda l: "".join([int_to_char[i] for i in l])
    
    # 2. Encode the start of the prompt
    start_ids = encode(input_text)
    prompt = (torch.tensor(start_ids, dtype=torch.long, device = device)[None, ...])
    
    # 3. Text generation
    with torch.no_grad():
        #with ctx:
        for i in range(num_samples):
            print(">> Generated legal text {}".format(i+1))
            res = model.generate(prompt, max_new_tokens = 500, temperature = 0.8, top_k = 200)
            print(decode(res[0].tolist()))
            print("\n")
            

# Legal text generator based on the RKLawGPT model
#input_example = "임대차계약"
input_example = input(">> Enter your start prompt :")
legal_text_generator(input_example, using_LLMs = False)