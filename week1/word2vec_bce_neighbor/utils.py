import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import wandb
import os
from val_data import validation_pairs

# Function to calculate cosine similarity between two word embeddings
def cosine_similarity_torch(embedding1, embedding2):
    embedding1 = embedding1.unsqueeze(0)  # Reshape to (1, embedding_dim)
    embedding2 = embedding2.unsqueeze(0)  # Reshape to (1, embedding_dim)
    cos_sim = F.cosine_similarity(embedding1, embedding2).item()
    return cos_sim

# Validation function to evaluate model on a set of word pairs
def validate_model(model, word_pairs, word_to_idx):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = len(word_pairs)
    
    for word1, word2, label in word_pairs:
        if word1 in word_to_idx and word2 in word_to_idx:
            word1_idx = torch.tensor([word_to_idx[word1]], dtype=torch.long)
            word2_idx = torch.tensor([word_to_idx[word2]], dtype=torch.long)
            
            # Calculate cosine similarity
            similarity = model(word1_idx, word2_idx)
            
            # Assume a threshold of 0.5 for being neighbors (adjustable)
            prediction = 1 if similarity > 0.5 else 0
            
            # Compare prediction with actual label (1 for similar, 0 for dissimilar)
            if prediction == label:
                correct += 1
    
    accuracy = correct / total
    return accuracy

def train_model(model, training_data, epochs, learning_rate, batch_size, word_to_idx):
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    # Convert training data to indices
    training_data_idx = [(word_to_idx[target], word_to_idx[context], label) for target, context, label in training_data]

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_data_idx)
        i = 0
        for batch_idx in range(0, len(training_data_idx), batch_size):
            batch = training_data_idx[batch_idx: batch_idx + batch_size]
            target_words, context_words, labels = zip(*batch)

            target_words = torch.tensor(target_words, dtype=torch.long)
            context_words = torch.tensor(context_words, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(target_words, context_words)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_mean = total_loss/(i+1)
            print(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(training_data_idx)//batch_size + 1}, Train Loss: {loss_mean:.4f}")
            i += 1
            if i % 1000 == 0:
                val_accuracy = validate_model(model, validation_pairs, word_to_idx)
                print(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(training_data_idx)//batch_size + 1}, Val Accuracy: {val_accuracy*100}\n")
                wandb.log({"val/acc": val_accuracy*100})
                wandb.log({"train/loss": loss_mean})
                # Save checkpoint at the end of each epoch
        checkpoint_path = os.path.join('week1', 'checkpoint', f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(training_data_idx),
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
# Example of getting similarity score for two words
def predict_neighbor(model, word1, word2, word_to_idx):
    word1_idx = torch.tensor([word_to_idx[word1]], dtype=torch.long)
    word2_idx = torch.tensor([word_to_idx[word2]], dtype=torch.long)
    probability = model(word1_idx, word2_idx).item()
    return probability
