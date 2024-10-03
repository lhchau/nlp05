import torch
from datasets import load_dataset
from data import *
from model import *
from utils import *
import wandb

# Hyperparameters
embedding_dim = 100  # Dimension of word embeddings
context_size = 2     # How many words before and after the target word to consider
epochs = 20           # Number of training epochs
learning_rate = 0.01
batch_size = 1024     # Batch size
negative_sampling_ratio = 1  # How many negative samples per positive pair

config = {
    'embedding_dim': embedding_dim,
    'context_size': context_size,
    'epochs': epochs,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'negative_sampling_ratio': negative_sampling_ratio
}
# current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(project='word2vec', name=f'emb_dim={embedding_dim}_con_size={context_size}_lr={learning_rate}_bs={batch_size}_neg_samp={negative_sampling_ratio}', config=config)


# Load Wikipedia dataset from Hugging Face
dataset = load_dataset('wikipedia', '20220301.simple', split='train')
dataset = dataset.select(range(10000))  # Optional: limit dataset size for quick training

# preprocess dataset
tokenized_corpus = tokenize_corpus(dataset)
word_to_idx, idx_to_word = build_vocab(tokenized_corpus)
vocab_size = len(word_to_idx)

training_data = generate_training_data(tokenized_corpus, context_size, negative_sampling_ratio, word_to_idx)

model = Word2VecBinary(vocab_size, embedding_dim)
train_model(model, training_data, epochs, learning_rate, batch_size, word_to_idx)