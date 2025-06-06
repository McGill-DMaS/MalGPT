import os
import pandas as pd
import re
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# Load the dataset
dataset = pd.read_csv('merged_features_with_explanation_v2.csv')

# Clean the labels: convert labels starting with 'benign_' to 'benign'
dataset['label'] = dataset['label'].str.replace(r'^benign_.+', 'benign', regex=True)

# Count the data in all categories
label_counts = dataset['label'].value_counts()
print("Label counts:")
print(label_counts)

# Train-test split
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Create folder with current date-time to save the split data
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'model_{current_datetime}'
os.makedirs(output_dir, exist_ok=True)

# Save the train and test data
train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

# Define Encoder-Decoder Transformer for Feature-to-Explanation
class Feature2ExplanationModel:
    def __init__(self, model_name='t5-small', max_input_len=512, max_output_len=150):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def train(self, features, explanations, batch_size=16, epochs=3):
        input_encodings = self.tokenizer(features.tolist(), truncation=True, padding=True, max_length=self.max_input_len, return_tensors='tf')
        output_encodings = self.tokenizer(explanations.tolist(), truncation=True, padding=True, max_length=self.max_output_len, return_tensors='tf')

        # Create a dataset from the encodings
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(input_encodings),
            output_encodings['input_ids']
        ))

        dataset = dataset.batch(batch_size)

        # Train the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                           loss=self.model.compute_loss)
        self.model.fit(dataset, epochs=epochs)

    def generate_explanation(self, feature_input):
        input_ids = self.tokenizer(feature_input, return_tensors="tf", max_length=self.max_input_len, truncation=True).input_ids
        output_ids = self.model.generate(input_ids, max_length=self.max_output_len)[0]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

# Prepare training data
train_features = train_data['features_column']  # Replace with the actual feature column name
train_explanations = train_data['Explanation']  # The target explanation column

# Initialize and train the model
model = Feature2ExplanationModel(model_name='t5-small', max_input_len=512, max_output_len=300)  # 300 tokens for 2-3 paragraphs
model.train(train_features, train_explanations, batch_size=8, epochs=5)

# Generate an explanation for one test data
test_feature = test_data.iloc[0]['features_column']  # Replace with the actual feature column name
generated_explanation = model.generate_explanation(test_feature)
print("Generated Explanation for one test data:")
print(generated_explanation)

