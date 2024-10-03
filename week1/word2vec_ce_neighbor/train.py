import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import random
from data import *
from model import *
from datasets import load_dataset


##############################
# Hyperparameters settings
##############################
embedding_dim = 256  # Dimension of word embeddings
context_size = 2     # How many words before and after the target word to consider
epochs = 10          # Number of training epochs
learning_rate = 0.001

##############################
# Corpus dataset
##############################
# Load Wikipedia dataset from Hugging Face
dataset = load_dataset('wikipedia', '20220301.en', split='train')

# Limit the dataset size for quick training (Optional, you can skip this to train on the full dataset)
# This will only take the first 10000 examples
dataset = dataset.select(range(10000))
##############################
# Tokenize corpus and create vocabulary
##############################

tokenized_corpus = tokenize_corpus(dataset)

word_to_idx, idx_to_word = build_vocab(tokenized_corpus)
vocab_size = len(word_to_idx)

training_data = generate_training_data(tokenized_corpus, context_size)

# Training the model
def train_model(model, training_data, epochs, learning_rate, batch_size):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    training_data_idx = [(word_to_idx[target], word_to_idx[context]) for target, context in training_data]

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_data_idx)
        
        for batch_idx in range(0, len(training_data_idx), batch_size):
            batch = training_data_idx[batch_idx: batch_idx + batch_size]
            target_words, context_words = zip(*batch)

            target_words = torch.tensor(target_words, dtype=torch.long)
            context_words = torch.tensor(context_words, dtype=torch.long)

            optimizer.zero_grad()
            output = model(target_words)
            loss = loss_fn(output, context_words)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(training_data_idx)//batch_size + 1}, Loss: {total_loss:.4f}")
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

model = Word2Vec(vocab_size, embedding_dim)
# Train the model
train_model(model, training_data, epochs, learning_rate, 2048)

# Example of getting word embedding
def get_word_embedding(word):
    word_idx = torch.tensor([word_to_idx[word]], dtype=torch.long)
    embedding = model.embeddings(word_idx).detach().numpy()
    return embedding

# Get embedding for the word "word2vec"
word_embedding = get_word_embedding("learning")
print(f"Embedding for 'word2vec': {word_embedding}")
