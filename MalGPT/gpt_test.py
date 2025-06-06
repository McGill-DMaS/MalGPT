import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Step 1: Load the pre-trained model
model = load_model('model_20241023_230754/encoder_decoder_model.h5')

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

# Step 4: Load X_test.csv and y_test.csv from the folder
X_test = pd.read_csv('model_20241023_230754/X_test.csv')
y_test = pd.read_csv('model_20241023_230754/y_test.csv')

# Convert the test data to NumPy arrays
X_test = X_test.values
y_test = y_test.values


# Tokenizer settings (adjust based on your original tokenizer)
# Assuming tokenizer was used during training, which converts y_test back to words
# tokenizer = ... (load or define the same tokenizer used during training)

# Step 5: Implement the decoding function
def decode_sequence(input_seq):
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

        # Get the token with the highest probability (greedy decoding)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        decoded_sentence.append(sampled_token_index)

        # Stop if we hit the stop token ('2') or exceed max length
        if sampled_token_index == 2 or len(decoded_sentence) > 100:  # Assuming max_len = 100
            stop_condition = True

        # Update the target sequence (of length 1) with the predicted token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


# Step 6: Test the model by generating explanations from the test set
for i in range(5):  # Generate explanations for the first 5 test examples
    input_seq = X_test[i:i + 1]  # Select one example from the test set
    decoded_explanation = decode_sequence(input_seq)

    # Convert token indices back to words using the tokenizer's inverse mapping
    # Assuming tokenizer was used during training
    # explanation_text = tokenizer.sequences_to_texts([decoded_explanation])[0]

    # Since we don't have the actual tokenizer loaded, we'll print the indices for now
   # print(f"Input features: {X_test[i]}")
    print(f"Generated explanation token indices: {decoded_explanation}")
    print(f"Actual explanation: {y_test[i]}")
    print("------------------------------------------------------")
