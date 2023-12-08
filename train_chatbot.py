import matplotlib.pyplot as plt
from prettytable import PrettyTable
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import SGD
import random
import re

def cleaning(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "what is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"\'ll", "will", sentence)
    sentence = re.sub(r"\'ve", "have", sentence)
    sentence = re.sub(r"\'re", "are", sentence)
    sentence = re.sub(r"\'d", "will", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"[-()\"#/@;:<>=|.?,]", "", sentence)
    return sentence

words = []
classes = []
documents = []
loss_values = []
accuracy_values = []
test_loss_values = []
test_accuracy_values = []
ignore_words = ['?', '!']

import json

with open("intents.json", encoding='utf-8') as file:
    intents = json.load(file)


lemmatizer = WordNetLemmatizer()

# Collect lemmatized words
lemmatized_words = [lemmatizer.lemmatize(w.lower()) for w in words]
lemmatized_words = sorted(list(set(lemmatized_words)))

# Count original words and their lemmatized forms
original_word_counts = [words.count(w) for w in lemmatized_words]
lemmatized_word_counts = [lemmatized_words.count(w) for w in lemmatized_words]

# Plotting the figure
plt.figure(figsize=(10, 6))
plt.barh(range(len(lemmatized_words)), original_word_counts, align='center', label='Original Words')
plt.barh(range(len(lemmatized_words)), lemmatized_word_counts, align='center', label='Lemmatized Words')
plt.yticks(range(len(lemmatized_words)), lemmatized_words)
plt.xlabel('Word Counts')
plt.ylabel('Words')
plt.title('Lemmatization: Original Words vs. Lemmatized Words')
plt.legend()
plt.tight_layout()
plt.show()

for intent in intents['intents']:
    classes.append(intent['tag'])
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(cleaning(pattern))
        words.extend(w)
        documents.append((w, intent['tag']))

words = [lemmatizer.lemmatize(w.lower()) for w in words]
words = sorted(list(set(words)))

print("Lemmatized words:")
print(words)
print("\nClasses:")
print(classes)
print("\nDocuments:")
for doc in documents:
    print("Tokenized words:", doc[0])
    print("Lemmatized words:", doc[1])
    print("Intent tag:", doc[2])
    print()

classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])

# Create testing data
test_data = training[:int(len(training)*0.2)]
test_x = np.array([x[0] for x in test_data])
test_y = np.array([x[1] for x in test_data])


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.00087, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



hist = model.fit(train_x, train_y, epochs=500, batch_size=5, verbose=1, validation_data=(test_x, test_y))
history = hist.history

model.save('chatbot_model.h5', hist)

print("Model created.")