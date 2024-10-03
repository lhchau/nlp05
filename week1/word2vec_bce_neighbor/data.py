import random
import wandb

# Tokenize corpus
def tokenize_corpus(dataset):
    tokenized_corpus = [sentence.split() for sentence in dataset['text']]
    return tokenized_corpus

# Build vocabulary
def build_vocab(tokenized_corpus):
    vocab = set(word for sentence in tokenized_corpus for word in sentence)
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}
    return word_to_idx, idx_to_word

# Generate positive and negative pairs for training
def generate_training_data(tokenized_corpus, context_size, negative_sampling_ratio, word_to_idx):
    data = []
    vocab_list = list(word_to_idx.keys())

    for sentence in tokenized_corpus:
        sentence_len = len(sentence)
        for i, word in enumerate(sentence):
            # Positive pairs (words in the same context window)
            for j in range(1, context_size + 1):
                if i - j >= 0:
                    data.append((word, sentence[i - j], 1))  # (target, context, label)
                if i + j < sentence_len:
                    data.append((word, sentence[i + j], 1))
            
            # Negative sampling (random words that are not neighbors)
            for _ in range(negative_sampling_ratio):
                negative_word = random.choice(vocab_list)
                while negative_word in sentence[max(0, i - context_size): min(i + context_size + 1, sentence_len)]:
                    negative_word = random.choice(vocab_list)  # Ensure it's not a neighbor
                data.append((word, negative_word, 0))  # (target, random_word, label)

    return data