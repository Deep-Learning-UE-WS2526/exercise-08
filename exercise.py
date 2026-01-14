import time
start_time = time.time()

import bz2

import numpy as np

from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers

def get_labels_and_texts(file, n=100000):
    labels = []
    texts = []
    i = 0
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
        i = i + 1
        if i >= n:
          return np.array(labels), texts
    return np.array(labels), texts

train_labels, train_texts = get_labels_and_texts("C:/Users/maris/Documents/Uni/IV/Deep Learning/exercise-08/data/train.ft.txt.bz2")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1
train_texts = tokenizer.texts_to_sequences(train_texts)

MAX_LENGTH = max(len(train_ex) for train_ex in train_texts)

train_texts = pad_sequences(train_texts, maxlen=MAX_LENGTH, padding="post")

from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, 
  random_state=0, test_size=0.1)

ffnn = models.Sequential()
ffnn.add(layers.Input(shape=(MAX_LENGTH,)))
ffnn.add(layers.Embedding(vocab_size, 200, input_length=MAX_LENGTH))
ffnn.add(layers.Flatten())
ffnn.add(layers.Dense(100, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(50, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(1, activation="sigmoid"))

ffnn.summary()

ffnn.compile(loss="binary_crossentropy", optimizer="sgd",
  metrics=["accuracy"])

ffnn.fit(train_texts, train_labels, epochs=10, batch_size=10, verbose=1)

from sklearn.metrics import accuracy_score
def evalu(mo):
  pred_labels = mo.predict(test_texts)
  pred_labels = [0 if x<0.5 else 1 for x in pred_labels]
  print(test_labels)
  print(pred_labels)
  print("accuracy: "+ str(accuracy_score(test_labels, pred_labels)))

evalu(ffnn)
print(str(time.time()-start_time) + " seconds")