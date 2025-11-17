# SMARTUAL MODEL TRAINING - T.I.P. Dataset
# Fine-tune SentenceTransformer on Q&A pairs for SmartUAL assistant

import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, models, losses
from torch.utils.data import DataLoader
import torch

# Configured and managed the files
DATA_FILE = "/content/TIP_QA_dataset_20000.csv"
MODEL_SAVE_PATH = "smartual_model"
PRETRAINED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
BATCH_SIZE = 32
EPOCHS = 3
MAX_LEN = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the data
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found!")

df = pd.read_csv(DATA_FILE)
print(f"Dataset loaded. Total rows: {len(df)}")

# Check required columns
if not all(col in df.columns for col in ['questionnaire', 'answer']):
    raise ValueError("CSV must contain 'questionnaire' and 'answer' columns")

# Prepareing the training
train_examples = [
    InputExample(texts=[str(row['questionnaire']).strip(), str(row['answer']).strip()])
    for _, row in df.iterrows()
    if str(row['questionnaire']).strip() and str(row['answer']).strip()
]

print(f"Total training examples: {len(train_examples)}")

# Model
word_embedding_model = models.Transformer(PRETRAINED_MODEL, max_seq_length=MAX_LEN)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

#Load Data
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# Training loss function
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Training the model
print("Starting training...")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=int(len(train_examples) * 0.1 / BATCH_SIZE),
    show_progress_bar=True
)

# Save the model for later use
model.save(MODEL_SAVE_PATH)
print(f"✅ SmartUAL model saved at '{MODEL_SAVE_PATH}'")

print(df.columns.tolist())
# simply checking the columns

import shutil

# Zip the folder for us to capture the overall model
shutil.make_archive("smartual_model", 'zip', "smartual_model")

from google.colab import files

# Download the ZIP file
files.download("smartual_model.zip")
