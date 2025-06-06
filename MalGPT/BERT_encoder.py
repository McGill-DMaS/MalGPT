import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification

# Step 1: Load the CSV data
file_path = 'merged_features_with_explanation-small.csv'  # Change to your CSV file path
data = pd.read_csv(file_path)

# Step 2: Drop unnecessary columns 'label', 'File Name'
data = data.drop(columns=['label', 'File Name'])

# Step 3: Check and clean the 'Explanation' column for non-string values
data['Explanation'] = data['Explanation'].fillna('').astype(str)

# Step 4: Split data into train and test sets (keeping 'Explanation' as target)
X = data.drop(columns=['Explanation'])  # Features
y = data['Explanation']  # Target

# Handle missing values in the feature set (X)
X = X.fillna(X.mean())

# Standardize features (optional)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Tokenize the text data for 'Explanation'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
y_train_encodings = tokenizer(y_train.tolist(), truncation=True, padding=True, max_length=200, return_tensors='tf')
y_test_encodings = tokenizer(y_test.tolist(), truncation=True, padding=True, max_length=200, return_tensors='tf')

# Step 6: Build a BERT-based model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Define a simple custom model that incorporates BERT for the encoder
input_features = tf.keras.Input(shape=(X_train.shape[1],), name='input_features')
dense_input = tf.keras.layers.Dense(768, activation='relu')(input_features)
dense_input = tf.expand_dims(dense_input, 1)  # Add time dimension

bert_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='bert_input')
bert_attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

bert_output = bert_model(bert_input, attention_mask=bert_attention_mask)[0]

# Add a dense layer for generating tokens
output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(tokenizer), activation='softmax'))(bert_output)

# Create the model
model = tf.keras.Model(inputs=[input_features, bert_input, bert_attention_mask], outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Step 7: Prepare training input
decoder_input_data = y_train_encodings['input_ids'][:, :-1]  # Remove the last token for decoder input
decoder_target_data = y_train_encodings['input_ids'][:, 1:]  # Shifted right for target

# Model training
model.fit(
    [X_train, decoder_input_data, y_train_encodings['attention_mask']],
    decoder_target_data,
    epochs=20,  # Adjust as needed
    batch_size=8,
    validation_split=0.2
)

# Step 8: Save the model and data splits
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
save_folder = f'model_{current_datetime}_BERT_encoder'

os.makedirs(save_folder, exist_ok=True)
model.save(os.path.join(save_folder, 'bert_encoder_decoder_model.h5'))

# Save the train/test splits
pd.DataFrame(X_train).to_csv(os.path.join(save_folder, 'X_train.csv'), index=False)
pd.DataFrame(X_test).to_csv(os.path.join(save_folder, 'X_test.csv'), index=False)
pd.DataFrame(y_train_encodings['input_ids'].numpy()).to_csv(os.path.join(save_folder, 'y_train.csv'), index=False)
pd.DataFrame(y_test_encodings['input_ids'].numpy()).to_csv(os.path.join(save_folder, 'y_test.csv'), index=False)
