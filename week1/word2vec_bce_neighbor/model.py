import torch
import torch.nn as nn

# Word2Vec model for binary classification (logistic task)
class Word2VecBinary(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecBinary, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, word1, word2):
        embed_word1 = self.embeddings(word1)
        embed_word2 = self.embeddings(word2)
        # Dot product between word embeddings (similarity)
        similarity = torch.sum(embed_word1 * embed_word2, dim=1)
        return torch.sigmoid(similarity)  # Output probability between 0 and 1
    
    def get_embedding(self, word):
        return self.embeddings(word)
