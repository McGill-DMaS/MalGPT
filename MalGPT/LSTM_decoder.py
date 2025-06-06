import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Step 1: Load the pre-trained model
model = load_model('model_20241026_211206_with_1000epoch/encoder_decoder_model.h5')

# Step 2: Extract the encoder model for inference
latent_dim = 512  # Latent dimension used in the trained model

# Encoder model: Extract the encoder inputs and states
encoder_inputs = model.input[0]  # Encoder input is the first input of the original model

# Get the LSTM layer from the model
encoder_lstm = model.get_layer('lstm')  # Get LSTM layer by name

# Extract encoder outputs and states (state_h, state_c)
encoder_outputs, state_h_enc, state_c_enc = encoder_lstm.output

# Create encoder inference model to return hidden states
encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

# Step 3: Create the decoder model for inference
# Unique names for decoder input states to avoid conflict with existing names
decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_state_input_h')
decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_state_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Reuse the decoder embedding and LSTM layers from the trained model
decoder_inputs = model.input[1]  # Decoder input from the original model
decoder_embedding = model.get_layer('embedding')(decoder_inputs)  # Use the original embedding layer
decoder_lstm = model.get_layer('lstm_1')  # Use the second LSTM layer from the original model

# Pass the embedding and the initial states to the LSTM
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)

# Reuse the Dense softmax layer
decoder_dense = model.get_layer('dense_1')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the inference decoder model with unique names
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + [state_h_dec, state_c_dec])

# Step 4: Implement the decoding function with debugging
# Step 4: Implement the decoding function with forced token generation
import numpy as np

def decode_sequence(input_seq, max_len=1000, min_tokens=10, temperature=1.0):
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)

    # Generate an empty target sequence of length 1 (start token is assumed to be '1')
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = 1  # '1' is assumed to be the start token

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        # Predict the next token using the decoder model
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Use sampling instead of greedy decoding to add randomness
        output_tokens = output_tokens[0, -1, :]  # Get the probability distribution
        output_tokens = np.asarray(output_tokens).astype('float64')
        output_tokens = np.log(output_tokens + 1e-9) / temperature  # Apply temperature
        exp_preds = np.exp(output_tokens)
        output_tokens = exp_preds / np.sum(exp_preds)

        # Sample the next token instead of taking the argmax
        sampled_token_index = np.random.choice(len(output_tokens), p=output_tokens)
        decoded_sentence.append(sampled_token_index)

        # Debug: print the predicted token and the current sentence length
        print(f"Predicted token: {sampled_token_index}, Current sentence length: {len(decoded_sentence)}")

        # Force the decoder to generate at least `min_tokens` before checking stop condition
        if len(decoded_sentence) > min_tokens:
            # Stop if we hit the stop token ('2') or exceed max length
            if sampled_token_index == 2 or len(decoded_sentence) >= max_len:
                stop_condition = True

        # Update the target sequence (of length 1) with the predicted token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

# Step 5: Load and prepare data (same as before, but tailored for test data)

# Load the test data
file_path = 'merged_features_with_explanation_v2.csv'  # Use your actual test data CSV file path
data = pd.read_csv(file_path)

# Drop unnecessary columns 'label', 'File Name'
data = data.drop(columns=['label', 'File Name'])

# Preprocess 'Explanation' column
data['Explanation'] = data['Explanation'].fillna('').astype(str)

# Split features and target
X = data.drop(columns=['Explanation'])  # Features
y = data['Explanation']  # Target (Explanation)

# Handle missing values in the features
X = X.fillna(X.mean())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prepare test data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Tokenize the 'Explanation' column
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(y_train)

# Convert 'Explanation' column to sequences
y_train_seq = tokenizer.texts_to_sequences(y_train)
y_test_seq = tokenizer.texts_to_sequences(y_test)

# Padding sequences
max_len = 100
y_train_pad = pad_sequences(y_train_seq, maxlen=max_len, padding='post')
y_test_pad = pad_sequences(y_test_seq, maxlen=max_len, padding='post')

# Step 6: Test the model by generating explanations from the test set
# Step 6: Test the model by generating explanations from the test set
for i in range(5):  # Generate explanations for the first 5 test examples
    input_seq = X_test[i:i + 1]  # Select one example from the test set
    decoded_explanation = decode_sequence(input_seq, max_len=1000, min_tokens=5)

    # Convert token indices back to words using the tokenizer's inverse mapping
    explanation_text = tokenizer.sequences_to_texts([decoded_explanation])[0]

    print(f"Generated explanation: {explanation_text}")
    print(f"Actual explanation: {y_test.iloc[i]}")
    print("------------------------------------------------------")

