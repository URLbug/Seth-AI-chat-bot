import pickle
import nltk
import string
import re

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, GRU
from keras.callbacks import ModelCheckpoint

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from __init__ import dataset


# Стоп слова на английском и функция для токинизации слов

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


# Токенизация данных

word_list = []

for i in dataset['intents']:
    for j in i['patterns']:
        word = tokinizators(j)
        word_list.append(word)

word_lists = sorted(set(word_list))

# Создаем файл со списком слов

pickle.dump(word_list, open(r'./dataset/word.pkl', 'wb'))

print('Word List - GOAL')

# Векторизируем данные

vectorizer = CountVectorizer()
list_vector = vectorizer.fit_transform(word_list)

X = list_vector.toarray()[:, 8:]
y = list_vector.toarray()[:, :8]

# Разделяем тренеровочные данные на test и на train
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2)

X_shape = X_train.shape
y_shape = y_train.shape

print(X_train.shape, y_train.shape)

# Создаем файлы с данными тренеровочных размерах

pickle.dump(X_shape, open(r'./dataset/X_shape.pkl', 'wb'))

print('X shape - GOAL')

# Создаем модель и обучаем его
# Модель будет обучаться на долговременной и кратковременной памяти GRU

model = Sequential()
model.add(Embedding(X_shape[0], 8, input_length=X_shape[1]))
model.add(GRU(857, input_shape=(X_shape[1], 1), return_sequences=True))
model.add(Dropout(.2))
model.add(GRU(857, return_sequences=True))
model.add(Dropout(.2))
model.add(GRU(128))
model.add(Dropout(.2))
model.add(Dense(y_shape[1],activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=2, save_best_only=True, mode='max')

desired_callbacks = [checkpoint]
seth = model.fit(X,y, epochs=55, callbacks=desired_callbacks, validation_split=.1, batch_size=10, steps_per_epoch=3, verbose=1)
model.save('Seth.model', seth)

print('Seth.modal')
