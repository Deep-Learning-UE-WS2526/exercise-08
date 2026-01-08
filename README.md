# Exercise 8: Embeddings

Deadline: 14.01.2026 23:59

This is the eighth exercise, and it is about input representation using embeddings.

## Step 1
Clone this repository to your local computer. On the command line, you would use the following command: `git clone https://github.com/Deep-Learning-UE-WS2526/exercise-08`.

## Step 2
Create a new branch in the repository, named after your UzK-account: `git checkout -b "UZKACCOUNT"`. 
We will re-use the same data set from [last weeks exercise](https://github.com/Deep-Learning-UE-WS2526/exercise-07). You can copy or move it into the data directory.

## Step 3: Preprocessing
The file `exercise.py` contains a template for this week. You first need to add code for the preprocessing: Use the keras-`Tokenizer` to tokenize the `train_texts`, convert it to an integer array and pad it to a fixed length. 


## Step 4: Embedding

The network as defined below should be ready to train. However, it uses the integer sequences from before directly as input features, which means we're missing out on generalization. Add an embedding layer to train embeddings. Don't forget to flatten after the embedding.

Results should be around 50% accuracy, which is disappointing for a binary classification task. The key to improve results here is a larger training set or leveraging pre-trained embeddings. Try out one of these options. If you opt for a larger training set, you might want to leave your computer running all night or even over the weekend. You can use the `n` parameter of the `get_labels_and_texts()`-function to control the size of the data set loaded.

## Step 6: Commit
Commit your changes to your local repository and push them to the server.