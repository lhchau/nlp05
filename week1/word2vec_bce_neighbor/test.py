import os
import torch
from model import Word2VecBinary
from val_data import validation_pairs
from datasets import load_dataset
from data import *
from utils import *


correct = 0
total = len(validation_pairs)

# Load Wikipedia dataset from Hugging Face
dataset = load_dataset('wikipedia', '20220301.simple', split='train')
dataset = dataset.select(range(1000))  # Optional: limit dataset size for quick training

# preprocess dataset
tokenized_corpus = tokenize_corpus(dataset)
word_to_idx, idx_to_word = build_vocab(tokenized_corpus)
vocab_size = len(word_to_idx)

model = Word2VecBinary(vocab_size=vocab_size, embedding_dim=100)
load_path = os.path.join('week1', 'checkpoint', 'model_epoch_25.pt')
checkpoint = torch.load(load_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set model to evaluation mode

for word1, word2, label in validation_pairs:
    if word1 in word_to_idx and word2 in word_to_idx:
        word1_idx = torch.tensor([word_to_idx[word1]], dtype=torch.long)
        word2_idx = torch.tensor([word_to_idx[word2]], dtype=torch.long)
        
        # Calculate cosine similarity
        similarity = model(word1_idx, word2_idx)
        
        # Assume a threshold of 0.5 for being neighbors (adjustable)
        prediction = 1 if similarity > 0.5 else 0
        print(word1, word2, label, prediction)
