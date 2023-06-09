![header](https://capsule-render.vercel.app/api?type=transparent&color=gradient&height=300&section=header&text=%20KRLawGPT%20&fontColor=317589&textBg=true&fontSize=100)

<img src="https://img.shields.io/badge/GPT-3776AB?style=flat-square&logo=Gitee&logoColor=white"/> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Colab-3776AB?style=flat-square&logo=Google Colab&logoColor=white"/> 

## Generative Pre-trained Transformer for producing Korean Legal Text

### 1. Model Description

 **Generative Pre-trained Transformer(GPT)** is a neural network-based language model trained on big data to produce human-like text. We have developed ***KRLawGPT*** specializes in legal texts. This language model uses a decoder-only transformer to generate expressive Korean legal text. ***KRLawGPT*** processes input text to perform both natural language generation and natural language processing to understand and generate legal text. 
 
  Our model is built to be pre-trained with its own GPT model or to leverage tokenizers and parameters from other GPT-based PLMs (GPT-2/3, KoGPT, etc.).
 ***KRLawGPT*** was pre-trained on a large-scale Korean legal dataset called CKLC(Clean Korean Legal Corpus). When given a small amount of prompt, it will generate large volumes of relevant and sophisticated judges-like Korean legal text.
 
 Moreover, the ***KRLawGPT python code*** we provided is designed not only for Korean legal texts, but also for users to train and optimize any of user own text data to generate related texts.


### 2. Model Usage


#### STEP 1. Load Text data and Build vacab

First step creates a split dataset (train.bin and val.bin) in that 'data' directory and builds a vocab. Then, it is ready to train KRLawGPT model.
```python
$ python model/vocab.py
```

If you want to utilize other GPT-based tokenizers, you have to set ```--using_LLMs = True```.
```python
$ python model/vocab.py --using_LLMs = True
```


#### STEP 2. Pre-train KRLawGPT on specific text data

This step saves the best performance model in validation dataset and creates KRLawGPT.pt and KRLawGPT_state_dict.pt in that 'output' directory. Now you can generate legal text with KRLawGPT.
```python
$ python model/train.py
```
If you want to leverage already trained GPT's parameters and weights from Hugging Face, you must set ```--using_LLMs = True``` and enter GPT-based pre-trained models name ```--model_type = 'kogpt' ```. Default is kogpt-2.
```python
$ python model/train.py --using_LLMs = True --model_type = 'kogpt'
```


#### STEP 3. Generate Legal Text

Lastly, enter the short words or sentences you want to generate. When given even a small number of words, pre-trained KRLawGPT will write large volumes of relevant and sophisticated judges-like Korean legal text.
```python
from model.generate_legal_text import *

input_text = input(">> Enter your start prompt :")
legal_text_generator(input_text)
```

#### Sample visualization

![generation](https://user-images.githubusercontent.com/105137667/231640382-a7129aa7-bf06-4b29-b767-f1fc3b42ccb5.gif)

### 4. Dev
- Seoul National University NLP Labs
- Navy Lee
