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

# Generate training data for skip-gram (target word, context word) = (center word, outside word)
def generate_training_data(tokenized_corpus, context_size): # context_size = window_size
    data = []
    for sentence in tokenized_corpus:
        sentence_len = len(sentence)
        for i, word in enumerate(sentence):
            for j in range(1, context_size + 1):
                if i - j >= 0:              # Check if word index in left side of center word is inside the sentence 
                    data.append((word, sentence[i - j]))
                if i + j < sentence_len:    # Check if word index in right side of center word is inside the sentence
                    data.append((word, sentence[i + j]))
    return data