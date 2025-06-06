import os
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime

# Load the dataset
dataset = pd.read_csv('merged_features_with_explanation.csv')

# Clean labels
dataset['label'] = dataset['label'].str.replace(r'^benign_.+', 'benign', regex=True)

# Prepare train/test split
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Prepare the folder for saving the model
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f'model_{current_datetime}'
os.makedirs(output_folder, exist_ok=True)

# Define the tokenizer and GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add a padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token


# Custom Dataset class
class ExplanationsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Combine features into a string as input
        input_text = f"File Name: {row['File Name']} Features: {row.drop(['File Name', 'Explanation', 'label']).to_dict()}"
        # Check if explanation is valid, if not, replace with a default string
        target_text = row['Explanation'] if isinstance(row['Explanation'], str) else "No explanation provided."
        # Tokenize input and target text
        inputs = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length,
                                return_tensors="pt")
        labels = self.tokenizer(target_text, truncation=True, padding='max_length', max_length=self.max_length,
                                return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels['input_ids'].flatten()
        }


# Create dataset and dataloader
train_dataset = ExplanationsDataset(train_data, tokenizer)
test_dataset = ExplanationsDataset(test_data, tokenizer)

# Use smaller batch size to fit within GPU memory
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# Define optimizer and loss
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Gradient accumulation steps
accumulation_steps = 4


# Training Loop
def train(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss = loss / accumulation_steps  # Normalize loss to accumulate gradients
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear CUDA memory
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch} | Loss: {avg_loss}")


# Train the model
for epoch in range(5):  # Train for 5 epochs
    train(model, train_loader, optimizer, device, epoch)

# Save the trained model
model.save_pretrained(output_folder)
tokenizer.save_pretrained(output_folder)
print(f"Model saved to {output_folder}")


# Prediction (Generating explanations)
def generate_explanation(features):
    model.eval()
    input_text = f"File Name: {features['File Name']} Features: {features.to_dict()}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate explanation
    output_ids = model.generate(input_ids, max_length=200, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


# Example to generate explanation from test data
example_features = test_data.iloc[0]
explanation = generate_explanation(example_features)
print("Generated Explanation:", explanation)
