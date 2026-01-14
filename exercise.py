import bz2

import numpy as np
import keras 

from keras import models, layers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers

#Das resultat aus diese Funktion ist ein String-Array bspw. [Ich bin hier,Du bist nicht hier, Und wo seid ihr gerade]
#Und ein Int Array mit labels
def get_labels_and_texts(file, n=10000):
    labels = []
    texts = []
    i = 0
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        #wir nehmen an jede Zeile beginnt mit __label__1, __label__2, __label__3. 
        #Mit (x[9]) greifen wir auf diese Zahl zu, weil es das 10. zeichen im String ist
        #Hier 1, 2, 3, 4, ...int wandelt das ganze in eine Zahl um und nimmt sie - 1, weil wir bei Labels mit 0 anfagen
        labels.append(int(x[9]) - 1) 
        #schneidet den String ab Stelle 10 ab und entfernt vorne und hinten alle Leerzeichen
        texts.append(x[10:].strip())
        i = i + 1 # bei jedem mal i erhöht, wenn i irgenwann so gr0ß ist wie n, also 10000 Zeilen, dann Ende und return
        if i >= n:
          return np.array(labels), texts
        #oder halt wenn die Datei weniger Zeilen hat, dann kann man auch return machen
    return np.array(labels), texts



train_labels, train_texts = get_labels_and_texts('data/train.ft.txt.bz2')




tokenizer = Tokenizer() 

 



#Tokenize train_texts
#convert to integer array and pad it to a fixed length
tokenizer = Tokenizer()
#hier wird text_to_word_sequence() implizit gemacht
#macht das array [Ich bin hier,Du bist da,Wir sind dort] in 
#[[Ich, bin, hier],[Du, bist, nicht, hier],[Und, wo, seid, ihr, gerade]]
tokenizer.fit_on_texts(train_texts) 
vocab_size = len(tokenizer.word_index) + 1

# Daraus machen wir jetzt Nummern, also [1,2,3,4],[5,6,7,8,9],.....
train_texts = tokenizer.texts_to_sequences(train_texts)

# max_lenght ist eine Konstante, die man immer groß schreibt, also MAX_LENGTH
# Hier wird das Maximum an Array Länge von den Expression also [1,2,3,4] in dem ganzen Array train_texts gesucht
MAX_LENGTH = max(len(train_ex) for train_ex in train_texts)

#Daraus machen wir eine Pad Sequence, damit das Modell lernen kann
#Und zwar[
# [Ich,   bin,  hier,   0,      0],
# [Du,    bist, nicht,  hier,   0],
# [Und,   wo,   seid,   ihr,    gerade]
#]
# Das sind aber jetzt keine Wörter mehr, sondern Zahlen
train_texts = pad_sequences(train_texts, maxlen=MAX_LENGTH, padding = "post")


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


