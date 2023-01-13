import pickle
import random

import numpy as np

from keras.models import load_model

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from __init__ import dataset


words = pickle.load(open(r'./dataset/words.pkl', 'rb'))
class_ = pickle.load(open(r'./dataset/class.pkl', 'rb'))

model = load_model('Seth.model')

stop_word = set(stopwords.words('english'))

# Токенизация текста 

def tokinizators(inputs: str) -> None:
    inputs = inputs.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(inputs)

    filtered = filter(lambda token: str(token) not in stop_word, tokens)
    return filtered

def seth(inputs: str):
    # Токенизация и векторизация данных
    
    send_word = tokinizators(inputs)

    bag = [0]*14

    for w in send_word:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    
    array = np.array(bag) 
    
    # Предсказываем, что написал пользователь

    result_array = model.predict(np.array([array]), verbose=0)[0]

    argmax = np.argmax(result_array)

    # Вывод данных

    results = dataset['intents'][argmax]['response']
    
    return results[random.randint(0, len(results)-1)]

