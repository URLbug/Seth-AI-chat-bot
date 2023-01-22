import pickle
import nltk
import random
import re
import string

import numpy as np

from keras.models import load_model

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from __init__ import dataset



X_shape = pickle.load(open(r'./dataset/X_shape.pkl', 'rb'))
word_list = pickle.load(open(r'./dataset/word.pkl', 'rb'))

model = load_model(r'Seth.model')

stop_word = set(stopwords.words('english'))

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_word = set(stopwords.words('english'))

def tokinizators(entry: str) -> None:
    tokens = entry.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return tokens

# Токенизация текста 

def seth(inputs: str) -> None:
    # Токенизация и векторизация данных
    
    send_word = tokinizators(inputs)

    all = [0]*len(X_shape[1])
    
    for w in send_word:
      for i, word in enumerate(word_list):
          if word == w:
            all[i] = 1
        
    array = np.array([all])
    print(array.shape)
    
    # Предсказываем, что написал пользователь

    result_array = model.predict(array, verbose=0)[0]

    argmax = np.argmax(result_array)

    # Вывод данных

    results = dataset['intents'][argmax]['response']
    
    return (results[random.randint(0, len(results)-1)], result_array, all)


