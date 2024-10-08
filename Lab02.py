# Import necessary libraries and modules
import json
import random
import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, GlobalAveragePooling1D, Dense


# 3. Prepare and load the data
# Define the path to the JSON file containing the intents and their patterns
file_path = r'D:\Download\phuong_intents.json'

# Initialize lists to store tags (intents), patterns (questions), responses, and contexts from the JSON file
tags = []
patterns = []
responses = []
contexts = []

# Open the JSON file and load its content
with open(file_path, 'r') as file:
    data = json.load(file)

# Loop through the loaded JSON data and populate the lists with the corresponding elements
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    for response in intent['responses']:
        responses.extend(response)
    contexts.append(intent.get('context', None))
    
    
# 4. Pre-processing
# Use LabelEncoder to convert the list of tags (intents) to numerical labels
label_encoder = LabelEncoder()
intent_labels = label_encoder.fit_transform(tags)
num_classes = np.max(intent_labels) + 1

# Tokenize the patterns: convert each pattern into a sequence of tokens
tokenizer = Tokenizer(num_words=800, oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, maxlen=45, padding='post')


# 5. Deep learning training
# Split the padded sequences and their labels into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, intent_labels, test_size=0.2, random_state=63)

# Define the model
model = Sequential([
    Embedding(input_dim=800, output_dim=25, input_length=45),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(10, activation='sigmoid'),
    Dense(8, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model using the training data and validate using the validation set
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_val, y_val), verbose=1)

# Train the model using the training data and validate using the validation set
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), verbose=1)

# Retrieve the accuracy from the training history and convert it to percentage
accuracy = history.history['accuracy'][-1]

# Print the accuracy of the model on the training data
print("Accuracy: {:.2f}%".format(accuracy * 100))


# 6. Testing the bot
# 1. Save Tokenizer
with open('D:\\Download\\tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 2. Save Encoder
with open('D:\\Download\\label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 3. Save Model
model.save('D:\\Download\\model')

# Load tokenizer
with open('D:/Download/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder
with open('D:/Download/label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Load model
model = tf.keras.models.load_model('D:/Download/model')

# Load intents JSON file
with open('D:/Download/phuong_intents.json') as file:
    intents = json.load(file)

# Define a function to predict the intent of a given text
def predict_intent(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=45, padding='post')
    pred = model.predict(padded, verbose=0)
    intent_index = np.argmax(pred)
    intent = label_encoder.inverse_transform([intent_index])[0]
    return intent

print("Chatbot is running. Type 'bye' to exit.")

# Loop for interacting with the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == "bye":
        print("Bot: Goodbye!")
        break
    intent = predict_intent(user_input)
    for item in intents['intents']:
        if item['tag'] == intent:
            print("Bot:", random.choice(item['responses']))
            break