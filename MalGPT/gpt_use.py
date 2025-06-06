import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd

# Load the saved model and tokenizer
model_dir = 'model_20241023_160530'
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Set device (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Add padding token to tokenizer
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS


# Function to generate explanations
def generate_explanation(features):
    model.eval()

    # Combine features into a single string
    input_text = f"File Name: {features['File Name']} Features: {features.to_dict()}"

    # Tokenize the input, truncate it to stay within the model's maximum length (1024 tokens)
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    # Check if the input exceeds the model's vocabulary size
    if torch.max(input_ids) >= model.config.vocab_size:
        raise ValueError("Input token exceeds model vocabulary size!")

    # Generate explanation with `max_new_tokens` instead of `max_length`
    output_ids = model.generate(
        input_ids,
        max_new_tokens=200,  # Generate up to 200 new tokens
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        attention_mask=input_ids.ne(tokenizer.pad_token_id)  # Set attention mask
    )

    # Decode the generated tokens into text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


# Example usage
test_data = pd.read_csv('merged_features_with_explanation.csv')  # Load your test dataset
example_features = test_data.iloc[0]  # Example row from the dataset
explanation = generate_explanation(example_features)
print("Generated Explanation:", explanation)
