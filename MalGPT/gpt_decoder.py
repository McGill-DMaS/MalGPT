import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer

# Reinitialize and fit the tokenizer on training data
tokenizer = Tokenizer(num_words=10000)
y_train = pd.read_csv('merged_features_with_explanation_v2.csv')['Explanation']  # Adjust path if needed
tokenizer.fit_on_texts(y_train)

# Load saved model and test data
model = load_model('model_20241025_163700_gpt_encoder_with_1000epoch/gpt_encoder_decoder_model.h5')

# Load the test set
X_test = pd.read_csv('model_20241025_105946_gpt_encoder/X_test.csv').values
y_test_pad = pd.read_csv('model_20241025_105946_gpt_encoder/y_test.csv').values

# Ensure you have the tokenizer loaded here, which was used during training.

# Parameters
latent_dim = 64
max_len = 100
start_token = 1  # Assuming start token index
stop_token = 2   # Assuming stop token index

# Step 1: Define Encoder Model for Inference
encoder_inputs = model.input[0]  # Encoder input from the original model
encoder_output = model.get_layer('tf.reshape').output  # Corrected layer name for reshaping
encoder_model = Model(encoder_inputs, [encoder_output, encoder_output])  # Encoder state

# Step 2: Define Decoder Model for Inference
decoder_inputs = model.input[1]  # Decoder input from the original model
decoder_embedding = model.get_layer('embedding')(decoder_inputs)
decoder_lstm = model.get_layer('lstm')
decoder_dense = model.get_layer('dense_9')  # Corrected layer name for dense layer

# Decoder state inputs for inference
decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_state_input_h')
decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_state_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Pass the embedding and the initial states to the LSTM
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_outputs = decoder_dense(decoder_outputs)

# Define the decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + [state_h, state_c])

# Step 3: Define the decoding function



def decode_sequence(input_seq, max_len=100, temperature=1.0):
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)

    # Initialize target sequence with the start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        # Predict the next token probabilities using the decoder model
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Get the token probabilities and adjust with temperature
        output_tokens = output_tokens[0, -1, :]  # Shape (vocab_size,)
        output_tokens = np.asarray(output_tokens).astype('float64')

        # Apply temperature
        output_tokens = np.log(output_tokens + 1e-10) / temperature
        exp_preds = np.exp(output_tokens)
        output_tokens = exp_preds / np.sum(exp_preds)

        # Sample the next token based on probabilities
        sampled_token_index = np.random.choice(len(output_tokens), p=output_tokens)
        decoded_sentence.append(sampled_token_index)

        # Exit condition: stop token or max length
        if sampled_token_index == stop_token or len(decoded_sentence) >= max_len:
            stop_condition = True

        # Update the target sequence (of length 1) with the predicted token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states for the next timestep
        states_value = [h, c]

    return decoded_sentence


# Step 4: Test the model with 10 samples from the test set
for i in range(10):
    input_seq = X_test[i:i+1]  # Take one test sample at a time
    decoded_explanation = decode_sequence(input_seq)

    # Convert token indices back to words
    explanation_text = tokenizer.sequences_to_texts([decoded_explanation])[0]
    actual_text = tokenizer.sequences_to_texts([y_test_pad[i]])[0]

    print(f"Generated explanation: {explanation_text}")
    print(f"Actual explanation: {actual_text}")
    print("------------------------------------------------------")
