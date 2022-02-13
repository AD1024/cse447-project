import torch
import torch.nn as nn
import tqdm
import numpy as np
from torch.nn import functional as F

from utils import cached, one_hot_vector, to_dictionary

@cached
def load_corpus():
    from nltk.corpus import brown
    vocab = brown.words()
    return list(map(lambda x: x.lower(), vocab[:len(vocab) // 20]))

class CharRNN(nn.Module):
    def __init__(self, vocab, n_ts=128, hidden_dim=128, num_layers=2, dropout=0.5) -> None:
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_ts = n_ts
        self.vocab = vocab

        self.drop_out = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(len(self.vocab),
                            self.hidden_dim,
                            self.num_layers,
                            batch_first=True,
                            dropout=self.dropout)
        self.act = nn.Linear(hidden_dim, len(self.vocab))

    def forward(self, x, h):
        out, (h, c)  = self.lstm(x, h)
        out = out.contiguous().view(out.size(0) * out.size(1), self.hidden_dim)
        act = self.act(out)
        return act, (h, c)
    
    def new_hidden(self, batch_size):
        h, c = torch.autograd.Variable(torch.randn(self.num_layers, batch_size, self.hidden_dim)),\
               torch.autograd.Variable(torch.randn(self.num_layers, batch_size, self.hidden_dim))
        h.zero_()
        c.zero_()
        return h, c

def to_batches(data, num_seq, seq_length):
    size_per_batch = num_seq * seq_length
    batch_size = len(data) // size_per_batch
    data = data[:batch_size * size_per_batch].reshape((num_seq, -1))
    yield data.shape[1] // seq_length
    for i in range(0, data.shape[1], seq_length):
        inp = data[:, i : i + seq_length]
        target = np.zeros_like(inp)
        target[:, :-1] = inp[:, 1:]
        if i + seq_length >= data.shape[1]:
            target[:, -1] = data[:, 0]
        else:
            target[:, -1] = data[:, i + seq_length]
        yield inp, target

def train(model: CharRNN, data, num_epoch, batch_size, seq_length=128, grad_clip=1, lr=0.001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_set = data[:int(0.9 * len(data))]
    epoch_progress = tqdm.tqdm(range(num_epoch))
    for epoch in epoch_progress:
        epoch_progress.set_description(f'Epoch {epoch}')
        hc = model.new_hidden(batch_size)
        batch_iter = to_batches(train_set, batch_size, seq_length)
        batch_seq_size = next(batch_iter)
        batch_progress = tqdm.tqdm(range(1, batch_seq_size + 1))
        for (i, (inp, target)) in enumerate(batch_iter):
            inp = torch.autograd.Variable(torch.from_numpy(one_hot_vector(inp, len(model.vocab))))
            target = torch.autograd.Variable(torch.from_numpy(target))
            hc = list(map(lambda x: torch.autograd.Variable(x.data), hc))
            model.zero_grad()
            out, h = model(inp, hc)
            loss = criterion(out, target.view(batch_size * seq_length))
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            optimizer.step()
            if i > 0 and i % 10 == 0:
                batch_progress.set_description('Batch {} | Loss: {:.4f}'.format(i, loss.item()))
        torch.save(model, 'char_rnn.pth')

def main():
    corpus = load_corpus()
    text = ' '.join(corpus)
    char2int, int2char = to_dictionary(text)
    vocab = list(char2int.keys())
    model = CharRNN(vocab)
    dataset = np.array([char2int[x] for x in text])
    train(model, dataset, 10, 10, grad_clip=5)

main()