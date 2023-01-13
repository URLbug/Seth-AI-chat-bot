import random
import pickle

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from __init__ import dataset


# Стоп слова на английском и функция для токинизации слов

stop_word = set(stopwords.words('english'))

def tokinizators(inputs: str) -> None:
    inputs = inputs.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(inputs)

    filtered = filter(lambda token: str(token) not in stop_word, tokens)
    return ' '.join(filtered)


# Токенизация данных

word_list = []
class_ = []
document = []

for i in dataset['intents']:
    for j in i['patterns']:
        word = tokinizators(j)
        word_list.append(word)
        
        document.append((word, i['tag']))

        if i['tag'] not in class_:
            class_.append(i['tag'])

word_lists = sorted(set(word_list))

# Создаем файлы с токенизированными данными

pickle.dump(word_lists, open(r'./dataset/words.pkl', 'wb'))
pickle.dump(class_, open(r'./dataset/class.pkl', 'wb'))

print('word list - GOAL')

# Создаем тренеровачный датасет для y

training = []
output_emple = [0] * len(class_)

for doc in document:
  bag = []
  word_pattern = doc[0]
  word_pattern = [tokinizators(i.lower()) for i in word_pattern]
  
  for word in word_list:
    bag.append(1) if word in word_pattern else bag.append(0)
  
  output_row = list(output_emple)
  output_row[class_.index(doc[1])] = 1
  
  training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

y_train = np.array(list(training[:,1]))

# Создаем тренеровочный датасет для X

countvectorizer = CountVectorizer(analyzer='word', stop_words='english')

count_wm = countvectorizer.fit_transform(word_list)

X_train = count_wm.toarray()

print(X_train.shape, y_train.shape)

# Создаем модель и обучаем его
# Модель будет обучаться на долговременной и кратковременной памяти LSTM

model = Sequential()
model.add(LSTM(256, input_shape=(14, 1), return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(128))
model.add(Dropout(.2))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

seth = model.fit(X_train, y_train, epochs=100, callbacks=desired_callbacks)
model.save('Seth.model', seth)

print('Seth.modal')
