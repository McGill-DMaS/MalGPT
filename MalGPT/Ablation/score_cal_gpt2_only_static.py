import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load and preprocess test data
file_path = 'data//merged_features_with_explanation-small-only-static.csv'
data = pd.read_csv(file_path)

# Replace specific benign labels and split data
data['label'] = data['label'].str.replace(r'^benign_.+', 'benign', regex=True)
X = data.drop(columns=['label', 'Explanation', 'File Name_x'])
y_actual = data['Explanation']
file_names = data['File Name_x']
label_names = data['label']

# Scale and split data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test, filenames_train, filenames_test, labelnames_train, labelnames_test = train_test_split(
    X_scaled, y_actual, file_names, label_names, test_size=0.2, random_state=42
)

# Initialize tokenizer on training data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(y_train)

# Load the model
model = load_model('model/model_20241114_233404_deep_gpt_encoder_small-only-static/gpt_encoder_decoder_model.h5')

# Set parameters
latent_dim = 1024
max_len = 100
start_token, stop_token = 1, 2

# Encoder model for inference
encoder_inputs = model.input[0]
encoder_output = model.get_layer('tf.reshape').output
encoder_model = Model(encoder_inputs, [encoder_output, encoder_output])

# Decoder model for inference
decoder_inputs = model.input[1]
decoder_embedding = model.get_layer('embedding')(decoder_inputs)
decoder_lstm = model.get_layer('lstm')
decoder_dense = model.get_layer('dense_9')

decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_state_input_h')
decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_state_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + [state_h, state_c])

# Decode function to generate explanations
def decode_sequence(input_seq, max_len=100, temperature=1.0):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token
    decoded_sentence = []

    while len(decoded_sentence) < max_len:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        output_tokens = np.log(output_tokens[0, -1, :] + 1e-10) / temperature
        output_tokens = np.exp(output_tokens) / np.sum(np.exp(output_tokens))
        sampled_token_index = np.random.choice(len(output_tokens), p=output_tokens)
        decoded_sentence.append(sampled_token_index)

        if sampled_token_index == stop_token or len(decoded_sentence) >= max_len:
            break

        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence

# Initialize scorer and results
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
results = []

# Generate explanations and evaluate
for i, (input_seq, actual_text, filename, labelname) in enumerate(
        zip(X_test, y_test, filenames_test, labelnames_test)):
    print(f"Processing item {i}")
    input_seq = input_seq.reshape(1, -1)
    decoded_tokens = decode_sequence(input_seq)
    generated_text = tokenizer.sequences_to_texts([decoded_tokens])[0]

    # Convert `actual_text_all` directly to text without indexing
    #actual_text = tokenizer.sequences_to_texts([actual_text_all])[0]
    # Calculate BLEU score, ROUGE score, and BERTScore
    bleu_score_value = sentence_bleu([actual_text.split()], generated_text.split())
    rouge_score_value = rouge.score(actual_text, generated_text)['rougeL'].fmeasure
    P, R, F1 = bert_score([generated_text], [actual_text], lang="en")
    bert_score_f1 = F1.mean().item() if hasattr(F1, 'mean') else np.mean(F1)

    # Append result for the current file and label
    results.append({
        'File Name': filename,
        'Label': labelname,
        'Actual Explanation': actual_text,
        'Generated Explanation': generated_text,
        'BLEU': bleu_score_value,
        'ROUGE': rouge_score_value,
        'BERTScore': bert_score_f1
    })

# Save results and print average scores
results_df = pd.DataFrame(results)
results_df.to_csv('result/evaluation_results-small_model_20241114_233404_deep_gpt_encoder_small-only-static.csv', index=False)

print("Average BLEU:", results_df['BLEU'].mean())
print("Average ROUGE-L:", results_df['ROUGE'].mean())
print("Average BERTScore:", results_df['BERTScore'].mean())
