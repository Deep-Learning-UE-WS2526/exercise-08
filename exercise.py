import bz2

import numpy as np

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


def get_labels_and_texts(file, n=10000):
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

# Use a larger subset of the data to improve generalization
train_labels, train_texts = get_labels_and_texts('data/train.ft.txt.bz2', n=20000)

# Tokenization and Padding
# Limit vocabulary size and sequence length to control model size
MAX_WORDS = 10000
MAX_LENGTH = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

sequences = tokenizer.texts_to_sequences(train_texts)
print(len(tokenizer.word_index))

padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH,padding='post', truncating='post')
print(padded_sequences.shape)


# Defining Vocabulary Size and Embedding Dimension
# Cap vocab at MAX_WORDS to ignore very rare terms
vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
print("Vocabulary Size: ", vocab_size)
embedding_dim = 100  # Typical size for simple text classification

# Add Embedding Layer to FFNN
ffnn = models.Sequential()
ffnn.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim))  # learnable word embeddings
ffnn.add(layers.Flatten())  # simple way to feed embeddings into Dense layers

# Hidden layers: ReLU + L2 weight decay + Dropout to reduce overfitting
ffnn.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(1, activation="sigmoid"))  # sigmoid for binary classification


ffnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])  # standard setup for binary labels

early_stopping = EarlyStopping(
    monitor="val_loss",       # stop when validation loss stops improving
    patience=2,                # allow a couple of bad epochs before stopping
    restore_best_weights=True, # keep the best model seen on validation data
)

ffnn.fit(
    padded_sequences,
    train_labels,
    epochs=15,       # upper bound; EarlyStopping will usually stop earlier
    batch_size=64,   # balanced trade-off between speed and stability
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping],
)

ffnn.summary()
