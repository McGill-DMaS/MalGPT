import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from datetime import datetime
import tensorflow as tf

# Step 1: Load the CSV data
file_path = 'merged_features_with_explanation-small.csv'  # Change to your CSV file path
data = pd.read_csv(file_path)

# Step 2: Drop unnecessary columns 'label', 'File Name'
data = data.drop(columns=['label', 'File Name'])

# Step 3: Check and clean the 'Explanation' column for non-string values
# Convert non-string values (e.g., floats, NaN) to empty strings
data['Explanation'] = data['Explanation'].fillna('').astype(str)

# Step 4: Split data into train and test sets (keeping 'Explanation' as target)
X = data.drop(columns=['Explanation'])  # Features
y = data['Explanation']  # Target

# Handle missing values in the feature set (X) by filling with the column mean
X = X.fillna(X.mean())

# Ensure all features are numeric (handle categorical variables if needed)
# If you have categorical features, you need to encode them. For now, assuming they are all numeric.

# Standardize features (optional but recommended)
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
max_len = 200  # Adjust based on your dataset
y_train_pad = pad_sequences(y_train_seq, maxlen=max_len, padding='post')
y_test_pad = pad_sequences(y_test_seq, maxlen=max_len, padding='post')

# Step 6: Build an Encoder-Decoder model
embedding_dim = 256
latent_dim = 512

# Encoder
encoder_inputs = Input(shape=(X_train.shape[1],))  # Input shape (batch_size, features)
encoder_dense = Dense(embedding_dim, activation='relu')(encoder_inputs)  # Dense layer to process the features

# Reshape the dense output to be compatible with LSTM input (batch_size, timesteps=1, features=embedding_dim)
encoder_reshape = tf.expand_dims(encoder_dense, 1)  # Add a time dimension (timesteps = 1)

# Now pass the reshaped input to the LSTM layer
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_reshape)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=10000, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(10000, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)



# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Step 7: Train the model

# Adjust the decoder input to be the same shape as the y_train_pad
decoder_input_data = np.zeros((y_train_pad.shape[0], max_len))  # Decoder input for training

# We will shift the target sequence by one timestep to create decoder inputs
decoder_input_data[:, 1:] = y_train_pad[:, :-1]  # Shifted target sequence

# Ensure the target sequence is correctly shaped for sparse categorical crossentropy
# In sparse categorical crossentropy, y_true must have the shape (batch_size, sequence_length)

# Model training
model.fit([X_train, decoder_input_data], y_train_pad, epochs=250, batch_size=64, validation_split=0.2)

# Step 8: Save the model and data splits
# Get the current datetime
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
save_folder = f'model_{current_datetime}_LSTM_eccoder_small_with_250epoch'

# Create folder if not exists
os.makedirs(save_folder, exist_ok=True)

# Save the model
model.save(os.path.join(save_folder, 'encoder_decoder_model.h5'))

# Save the train/test splits
pd.DataFrame(X_train).to_csv(os.path.join(save_folder, 'X_train.csv'), index=False)
pd.DataFrame(X_test).to_csv(os.path.join(save_folder, 'X_test.csv'), index=False)
pd.DataFrame(y_train_pad).to_csv(os.path.join(save_folder, 'y_train.csv'), index=False)
pd.DataFrame(y_test_pad).to_csv(os.path.join(save_folder, 'y_test.csv'), index=False)

# # Step 9: Inference - Define the encoder and decoder models for generating explanations
#
# # Encoder Model for Inference
# encoder_model = Model(encoder_inputs, encoder_states)
#
# # Decoder Model for Inference
# decoder_state_input_h = Input(shape=(latent_dim,))
# decoder_state_input_c = Input(shape=(latent_dim,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#
# decoder_embedding2 = decoder_embedding(decoder_inputs)
# decoder_lstm_outputs, state_h2, state_c2 = decoder_lstm(decoder_embedding2, initial_state=decoder_states_inputs)
# decoder_states2 = [state_h2, state_c2]
# decoder_outputs2 = decoder_dense(decoder_lstm_outputs)
#
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs2] + decoder_states2
# )
#
#
# # Function to decode the sequence
# def decode_sequence(input_seq):
#     # Encode the input as state vectors.
#     states_value = encoder_model.predict(input_seq)
#
#     # Generate empty target sequence of length 1 (start with the start token, assuming it's index 1)
#     target_seq = np.zeros((1, 1))
#     target_seq[0, 0] = 1  # This assumes '1' is the start token
#
#     # Sampling loop to generate the output sequence
#     stop_condition = False
#     decoded_sentence = []
#
#     while not stop_condition:
#         output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
#
#         # Get the token with the highest probability (this is greedy decoding, adjust for beam search if needed)
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         decoded_sentence.append(sampled_token_index)
#
#         # Exit condition: either hit max length or find the stop token.
#         if sampled_token_index == 2 or len(decoded_sentence) > max_len:  # Assuming '2' is the stop token
#             stop_condition = True
#
#         # Update the target sequence (of length 1).
#         target_seq = np.zeros((1, 1))
#         target_seq[0, 0] = sampled_token_index
#
#         # Update states
#         states_value = [h, c]
#
#     return decoded_sentence
#
#
# # Step 10: Test the model by generating explanations from the test set
#
# # Convert test features to the appropriate format
# # This depends on how the features are structured, here assuming a sequence format
# for i in range(5):  # Generate explanations for 5 test examples
#     input_seq = X_test[i:i + 1]  # Select one example from the test set
#     decoded_explanation = decode_sequence(input_seq)
#
#     # Convert token indices back to words using the tokenizer's inverse mapping
#     explanation_text = tokenizer.sequences_to_texts([decoded_explanation])[0]
#
#     print(f"Input features: {X_test[i]}")
#     print(f"Generated explanation: {explanation_text}")
#     print(f"Actual explanation: {y_test.iloc[i]}")
#     print("------------------------------------------------------")
