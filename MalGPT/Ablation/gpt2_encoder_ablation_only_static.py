import os
import pandas as pd
import numpy as np
import  pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, MultiHeadAttention, LayerNormalization, Dropout, Add
from datetime import datetime
import tensorflow as tf

# Step 1: Load the CSV data
file_path = 'data/merged_features_with_explanation-small-only-static.csv'  # Change to your CSV file path
data = pd.read_csv(file_path)

# Step 2: Drop unnecessary columns 'label', 'File Name'
data = data.drop(columns=['label', 'File Name_x'])

# Step 3: Clean the 'Explanation' column for non-string values
data['Explanation'] = data['Explanation'].fillna('').astype(str)

# Step 4: Split data into train and test sets (keeping 'Explanation' as target)
X = data.drop(columns=['Explanation'])  # Features
y = data['Explanation']  # Target

# Handle missing values in the feature set (X) by filling with the column mean
X = X.fillna(X.mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Tokenize the text data for 'Explanation'
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(y_train)

# Convert the 'Explanation' column to sequences
y_train_seq = tokenizer.texts_to_sequences(y_train)
y_test_seq = tokenizer.texts_to_sequences(y_test)

# Padding the sequences to have the same length
max_len = 200
y_train_pad = pad_sequences(y_train_seq, maxlen=max_len, padding='post')
y_test_pad = pad_sequences(y_test_seq, maxlen=max_len, padding='post')

# GPT-style Transformer Block
def gpt_block(x, num_heads, key_dim, ff_dim, dropout_rate=0.1):
    # Self-attention layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = Add()([x, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    # Feed-forward network
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(key_dim)(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Add()([attention_output, ff_output])
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output)

    return ff_output

# Parameters for Transformer
num_heads = 32
key_dim = 1024
ff_dim = 1024
num_layers = 32
dropout_rate = 0.1

# Step 6: Build GPT-Style Encoder
encoder_inputs = Input(shape=(X_train.shape[1],))  # Input shape (batch_size, features)
encoder_dense = Dense(key_dim, activation="relu")(encoder_inputs)  # Project input to key_dim size
encoder_dense_expanded = tf.expand_dims(encoder_dense, 1)  # Add time dimension (1 timestep)

# Transformer blocks
x = encoder_dense_expanded
for _ in range(num_layers):
    x = gpt_block(x, num_heads, key_dim, ff_dim, dropout_rate)

# Flatten the output of transformer
encoder_output = tf.reshape(x, [-1, key_dim])  # Reshape to (batch_size, key_dim)
encoder_states = [encoder_output, encoder_output]  # State placeholder for decoder (simulate LSTM states)

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=10000, output_dim=key_dim)(decoder_inputs)
decoder_lstm = LSTM(key_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(10000, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Step 7: Train the model
decoder_input_data = np.zeros((y_train_pad.shape[0], max_len))  # Decoder input for training
decoder_input_data[:, 1:] = y_train_pad[:, :-1]  # Shifted target sequence

# Model training
model.fit([X_train, decoder_input_data], y_train_pad, epochs=10, batch_size=64, validation_split=0.2)

# Step 8: Save the model and data splits
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
save_folder = f'model/model_{current_datetime}_deep_gpt_encoder_small-only-static'

# Create folder if not exists
os.makedirs(save_folder, exist_ok=True)

# Save the model
model.save(os.path.join(save_folder, 'gpt_encoder_decoder_model.h5'))
# Save the tokenizer
with open(os.path.join(save_folder, 'tokenizer.pkl'), 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Save the train/test splits
pd.DataFrame(X_train).to_csv(os.path.join(save_folder, 'X_train.csv'), index=False)
pd.DataFrame(X_test).to_csv(os.path.join(save_folder, 'X_test.csv'), index=False)
pd.DataFrame(y_train_pad).to_csv(os.path.join(save_folder, 'y_train.csv'), index=False)
pd.DataFrame(y_test_pad).to_csv(os.path.join(save_folder, 'y_test.csv'), index=False)
