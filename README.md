![header](https://capsule-render.vercel.app/api?type=transparent&color=gradient&height=300&section=header&text=%20KRLawGPT%20&fontColor=317589&textBg=true&fontSize=100)

	<img src="https://github-readme-stats.vercel.app/api?username=Navy10021&show_icons=true">

## Generative Pre-trained Transformer for producing Korean Legal Text

### 1. Model Description

 Generative Pre-trained Transformer(GPT) is a neural network-based language model trained on big data to produce human-like text. We have developed ***KRLawGPT*** specializes in legal texts. This language model uses a decoder-only transformer to generate expressive Korean legal text. ***KRLawGPT*** processes input text to perform both natural language generation and natural language processing to understand and generate legal text. 
 
  Our model is built to be pre-trained with its own GPT model or to leverage tokenizers and parameters from other GPT-based PLMs (GPT-2/3, KoGPT, etc.).
 ***KRLawGPT*** was pre-trained on a large-scale Korean legal dataset called CKLC(Clean Korean Legal Corpus). When given a small amount of prompt, it will generate large volumes of relevant and sophisticated judges-like Korean legal text.
 
 Moreover, the ***KRLawGPT python code*** we provided is designed not only for Korean legal texts, but also for users to train and optimize any of their own text data to generate related texts.


### 2. Model Usage


#### STEP 1. Load Text data and Build vacab

```python
$ python model/vocab.py
```

If you want to utilize other GPT-based tokenizers, you must set ```--using_LLMs = True```.
```python
$ python model/vocab.py --using_LLMs = True
```
This step creates a train.bin and val.bin in that 'data' directory and builds a vocab. Now it is ready to train KRLawGPT model.


#### STEP 2. Pre-train KRLawGPT on specific text data

```python
$ python model/train.py
```
If you want to leverage already trained GPT's parameters and weights from Hugging Face, you must set ```--using_LLMs = True```.
```python
$ python model/train.py --using_LLMs = True
```
This step saves the best performance model in validation and creates KRLawGPT.pt and KRLawGPT_state_dict.pt in that 'output' directory.


#### STEP 3. Generate Legal Text
Enter the short words or sentence you want to generate. When given even a small number of words, KRLawGPT will generate large volumes of relevant and sophisticated judges-like Korean legal text.

```python
from model.generate_legal_text import *

input_text = input(">> Enter your start prompt :")
legal_text_generator(input_text)
```

Generation GIF
