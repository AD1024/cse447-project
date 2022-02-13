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
    return list(map(lambda x: x.lower(), vocab[:len(vocab) // 10]))

class CharRNN(nn.Module):
    def __init__(self, vocab, int2char, char2int, n_ts=128, hidden_dim=512, num_layers=2, dropout=0.5) -> None:
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_ts = n_ts
        self.vocab = vocab
        self.int2char = int2char
        self.char2int = char2int

        self.drop_out = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(len(self.vocab),
                            self.hidden_dim,
                            self.num_layers,
                            batch_first=True,
                            dropout=self.dropout)
        self.act = nn.Linear(hidden_dim, len(self.vocab))

    def forward(self, x, h):
        out, (h, c)  = self.lstm(x, h)
        out = self.drop_out(out)
        out = out.contiguous().view(out.size(0) * out.size(1), self.hidden_dim)
        act = self.act(out)
        return act, (h, c)

    def predict(self, characters, h=None, num_choice=1):
        if h is None:
            h = self.new_hidden(1)
        inp = np.array([[self.char2int[x] for x in characters]])
        inp = torch.autograd.Variable(torch.from_numpy(one_hot_vector(inp, len(self.vocab))))
        out, h = self.forward(inp, h)
        prob = F.softmax(out[-1], dim=0)
        prob, ch = prob.topk(num_choice)
        prob = prob.numpy()
        ch = ch.numpy()
        ans = np.random.choice(ch, p=prob / prob.sum())
        return self.int2char[ans], h
    
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
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_set = data[:int(0.9 * len(data))]
    epoch_progress = tqdm.tqdm(range(num_epoch), position=1)
    for epoch in epoch_progress:
        epoch_progress.set_description(f'Epoch {epoch}')
        hc = model.new_hidden(batch_size)
        batch_iter = to_batches(train_set, batch_size, seq_length)
        batch_seq_size = next(batch_iter)
        batch_progress = tqdm.tqdm(range(1, batch_seq_size + 1), position=0)
        for (i, (inp, target)) in zip(batch_progress, batch_iter):
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

def test(model: CharRNN, sentence, char2int, int2char, predict_length=512):
    print("testing")
    with torch.no_grad():
        model.eval()
        sentence = sentence.lower()
        # print(f'Begin: {sentence}')
        h = model.new_hidden(1)
        for ch in sentence:
            c, h = model.predict(ch, h)
        result = []
        result.append(c)
        for _ in range(predict_length):
            c, h = model.predict(result[-1], h, num_choice=3)
            result.append(c)
        return ''.join(result)

def main():
    corpus = load_corpus()
    text = ' '.join(corpus)

    # char2int, int2char = to_dictionary(text)
    # vocab = list(char2int.keys())
    # model = CharRNN(vocab, int2char, char2int)
    # dataset = np.array([char2int[x] for x in text])
    # train(model, dataset, 20, 10, grad_clip=5, lr=0.002)

    model = torch.load('char_rnn_v1.pth')
    print(test(model, 'happy new ye', model.char2int, model.int2char, predict_length=1024))
    print(text[-2])

main()
