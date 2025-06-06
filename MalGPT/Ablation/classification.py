import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from datetime import datetime

# Step 1: Load the CSV data
file_path = 'data/merged_features_with_explanation-small-without-CEG.csv'
data = pd.read_csv(file_path)

# Step 2: Prepare and process labels with LabelEncoder and print label encoding scheme
data['label'] = data['label'].str.replace(r'^benign_.+', 'benign', regex=True)
label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['label'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

print("Label Encoding Scheme:")
for label, index in label_mapping.items():
    print(f"{label}: {index}")

# Save the encoding scheme to a file
with open('label_encoding_scheme.txt', 'w') as f:
    for label, index in label_mapping.items():
        f.write(f"{label}: {index}\n")

# Prepare features and labels
X = data.drop(columns=['label', 'Explanation', 'File Name_x', 'label_encoded'])
y = pd.get_dummies(data['label_encoded']).values

print("\nOverall Label Distribution:")
print(data['label'].value_counts())

# Step 3: Scale features and split the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Print label counts in train and test sets
print("\nLabel Distribution in Training Set:")
print(pd.Series(np.argmax(y_train, axis=1)).value_counts())
print("\nLabel Distribution in Test Set:")
print(pd.Series(np.argmax(y_test, axis=1)).value_counts())

# Step 4: Build neural network with L2 regularization and dropout
input_layer = Input(shape=(X_train.shape[1],))
dense_layer = Dense(8, activation='relu', kernel_regularizer=l2(0.015))(input_layer)
dense_layer = Dropout(0.5)(dense_layer)
output_layer = Dense(y.shape[1], activation='softmax')(dense_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=2, batch_size=16, validation_split=0.2)

# Step 6: Evaluate the model and print metrics
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_labels, y_pred_labels)
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
class_report = classification_report(y_test_labels, y_pred_labels)
roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')

print("\nTest Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print("\nROC AUC Score:", roc_auc)

# Save model metrics and label encoding
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
save_folder = f'model/model_{current_datetime}-without-CEG'
os.makedirs(save_folder, exist_ok=True)
pd.DataFrame(conf_matrix).to_csv(os.path.join(save_folder, 'confusion_matrix.csv'))

with open(os.path.join(save_folder, 'metrics_report.txt'), 'w') as f:
    f.write(f"Test Accuracy: {accuracy}\n")
    f.write(f"Classification Report:\n{class_report}\n")
    f.write(f"ROC AUC Score: {roc_auc}\n")
    f.write("\nLabel Encoding Scheme:\n")
    for label, index in label_mapping.items():
        f.write(f"{label}: {index}\n")

print("\nModel, metrics, and label encoding scheme saved successfully.")
