import string
import os

import nltk
import numpy as np
from tensorflow import keras

#building the vocabulary
def collect_all_words():
    all_words = set()
    for root, dirs, files in os.walk("./reviewsTrain"):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                file_content = f.readline()
                tokens = nltk.word_tokenize(file_content)
                tokens = [token for token in tokens if token not in string.punctuation]
                for token in tokens:
                    all_words.add(token)
    return all_words

all_words = collect_all_words()
all_words.add("<UNK>")
all_words = list(all_words)
word_indices = {}
for i, word in enumerate(all_words):
    word_indices[word] = i


#pre-processing the training reviews.
one_hot_representations = []
y_labels = []

def preprocess_data(filepath):
    for root, dirs, files in os.walk(filepath):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                template = np.zeros(len(all_words))
                file_content = f.readline()
                tokens = nltk.word_tokenize(file_content)
                tokens = [token for token in tokens if token not in string.punctuation]
                for token in tokens:
                    try:
                        template[word_indices[token]] = 1
                    except:
                        template[word_indices["<UNK>"]] = 1
                one_hot_representations.append(template)
                review = f.readline()
                if review == "pro-Lebron":
                    y_labels.append(0)
                else:
                    y_labels.append(1)
    x_train = np.array(one_hot_representations)
    y_train = np.array(y_labels)
    return (x_train, y_train)

vocabulary_size = len(all_words)
embedding_layer_size = 16

#building and training the model
def build_model(vocabulary_size, embedding_layer_size):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_layer_size))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(2, activation="softmax"))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model(vocabulary_size, embedding_layer_size)
x_train, y_train = preprocess_data("./reviewsTrain")
model.fit(x_train, y_train, epochs=10, batch_size=4)

#Testing the Model
x_test, y_test = preprocess_data("./reviewTest")
result = np.argmax(model.predict(x_test))
if result == 0:
    print("This is a pro-Lebron review")
else:
    print("This is an anti-Lebron review")



