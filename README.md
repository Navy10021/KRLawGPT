![header](https://capsule-render.vercel.app/api?type=transparent&color=gradient&height=300&section=header&text=%20KRLawGPT%20&fontColor=317589&textBg=true&fontSize=100)

<img src="https://img.shields.io/badge/GPT-3776AB?style=flat-square&logo=Gitee&logoColor=white"/> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Colab-3776AB?style=flat-square&logo=Google Colab&logoColor=white"/> 

## Generative Pre-trained Transformer for producing Korean Legal Text

### Abstract :
In this work, we introduce the development and application of a **Generative Pre-trained Transformer (GPT)** tailored for producing Korean legal text, named ***KRLawGPT***. As a neural network-based language model, ***KRLawGPT*** is designed to generate expressive and relevant Korean legal text through a decoder-only transformer. This model is pre-trained on a comprehensive Korean legal dataset, CKLC (Clean Korean Legal Corpus), and is equipped to handle both natural language generation and natural language processing tasks. The thesis also outlines the model's adaptability for training on user-specific text data, broadening its utility beyond the realm of legal texts.


### 1. Model Description

## 1.1. Generative Pre-trained Transformer (GPT) for Legal Texts
 ***KRLawGPT*** is introduced as a language model specifically crafted for the generation of Korean legal text. Utilizing a decoder-only transformer, this model is trained on a large-scale legal dataset, CKLC, enabling it to generate human-like and sophisticated legal texts. ***KRLawGPT*** stands out for its capability to process input text, performing both natural language generation and processing tasks.

## 1.2. Model Flexibility and Integration
  The model is built with flexibility in mind, allowing users to either pre-train it with its own GPT model or leverage tokenizers and parameters from other GPT-based Pre-trained Language Models (PLMs) such as GPT-2/3 or KoGPT. Moreover, ***KRLawGPT*** supports training and optimization on user-provided text data, extending its functionality beyond the legal domain.


### 2. Model Usage


#### STEP 1. Loading Text Data and Building Vocabulary

The initial step involves creating a split dataset (train.bin and val.bin) in the 'data' directory and building a vocabulary. Users can set options to utilize other GPT-based tokenizers for added versatility.
```python
$ python model/vocab.py
```

If you want to utilize other GPT-based tokenizers, you have to set ```--using_LLMs = True```.
```python
$ python model/vocab.py --using_LLMs = True
```


#### STEP 2. Pre-training KRLawGPT on Specific Text Data

This step involves training the ***KRLawGPT*** model, saving the best-performing model on the validation dataset, and generating KRLawGPT.pt and KRLawGPT_state_dict.pt in the 'output' directory. Users have the option to leverage pre-trained models from Hugging Face by setting specific parameters.
```python
$ python model/train.py
```
If you want to leverage already trained GPT's parameters and weights from Hugging Face, you must set ```--using_LLMs = True``` and enter GPT-based pre-trained models name ```--model_type = 'kogpt' ```. Default is kogpt-2.
```python
$ python model/train.py --using_LLMs = True --model_type = 'kogpt'
```


#### STEP 3. Generate Legal Text

Users can input short words or sentences to generate large volumes of relevant and sophisticated judges-like Korean legal text using the pre-trained ***KRLawGPT*** model.
```python
from model.generate_legal_text import *

input_text = input(">> Enter your start prompt :")
legal_text_generator(input_text)
```

#### Sample visualization

![generation](https://user-images.githubusercontent.com/105137667/231640382-a7129aa7-bf06-4b29-b767-f1fc3b42ccb5.gif)

### 3. Development
- Seoul National University NLP Labs
- Under the guidance of Navy Lee
