import torch
import torch.nn as nn
import tqdm
import argparse
import numpy as np
from torch.nn import functional as F
# from datasets import load_dataset
import time
import os
import random

from utils import cached, one_hot_vector, to_dictionary

def load_comments(filename='1.txt'):
    result = []
    with open(filename, encoding='UTF-8') as fp:
        for x in fp.readlines():
            if x == '\n' or len(x) == 0:
                continue
            result.append(x)
            result.append('。')
    return result

@cached
def load_corpus():
    from nltk.corpus import brown
    vocab = brown.words()
    return list(map(lambda x: x.lower(), vocab[:len(vocab)]))

@cached
def load_wiki():
    res = []
    progress = tqdm.tqdm(list(os.listdir('data')))
    for fname in progress:
        progress.set_description('loading data/' + fname)
        with open('data/' + fname) as f:
            s = f.read()
            res.append(s[:160000])
    return res

class CharRNN(nn.Module):
    def __init__(self, vocab, char2int, int2char, n_ts=128, embedding_dim=300, hidden_dim=512, num_layers=3, dropout=0.3) -> None:
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_ts = n_ts
        self.vocab = vocab
        self.char2int = char2int
        self.int2char = int2char

        self.drop_out = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(len(self.vocab) + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            self.num_layers,
                            batch_first=True,
                            dropout=self.dropout)
        self.act = nn.Linear(hidden_dim, len(self.vocab) + 1)

    def forward(self, x, h):
        out = self.drop_out(self.embedding(x))
        out, (h, c)  = self.lstm(out, h)
        out = out.contiguous().view(out.size(0) * out.size(1), self.hidden_dim)
        act = self.act(out)
        return act, (h.detach(), c.detach())

    def predict(self, characters, h=None, num_choice=3):
        if h is None:
            h = self.new_hidden(1)
        inp = torch.from_numpy(np.array([[self.char2int[x] if x in self.char2int else 0 for x in characters]], dtype=np.longlong))
        # inp = torch.autograd.Variable(torch.from_numpy(one_hot_vector(inp, len(self.vocab)+1)))
        if torch.cuda.is_available():
            inp = inp.cuda()
        out, h = self.forward(inp, h)
        prob = F.softmax(out[-1], dim=0)
        prob, ch = prob.topk(num_choice)
        prob = prob.cpu().numpy()
        ch = ch.cpu().numpy()
        return [self.int2char[x] for x in ch], h
    
    def new_hidden(self, batch_size):
        w = next(self.parameters())
        h, c = w.new_zeros((self.num_layers, batch_size, self.hidden_dim)),\
               w.new_zeros((self.num_layers, batch_size, self.hidden_dim))
        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()
        return h, c

def to_batches(data, num_seq, seq_length, volcab_len):
    size_per_batch = num_seq * seq_length
    num_batch = len(data) // size_per_batch
    data = data[:num_batch * size_per_batch].reshape((num_batch, -1))
    print(f'Data size: {data.shape}')
    inputs = []
    targets = []
    print(f'Total inputs: {data.shape[1]}')
    for i in range(0, data.shape[1], seq_length):
        if i % 1000 == 0:
            print(f'Processing Inputs: {i}')
        inp = data[:, i : i + seq_length]
        target = np.zeros_like(inp)
        target[:, :-1] = inp[:, 1:]
        if i + seq_length >= data.shape[1]:
            target[:, -1] = data[:, 0]
        else:
            target[:, -1] = data[:, i + seq_length]  
        # inputs.append(one_hot_vector(inp, volcab_len))
        inputs.append(inp.astype(np.longlong))
        targets.append(target)
    # print(np.concatenate(inputs).shape)
    return num_batch, \
        torch.autograd.Variable(torch.from_numpy(np.concatenate(inputs).reshape(num_batch, -1, seq_length))), \
        torch.autograd.Variable(torch.from_numpy(np.array(targets).reshape(num_batch, -1, seq_length)))

def train(model: CharRNN, data, num_epoch, batch_size, checkpoint_filename, seq_length=128, grad_clip=1, lr=0.001):
    start_time = time.time()
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_set = data[:int(0.9 * len(data))]
    epoch_progress = tqdm.tqdm(range(num_epoch), position=0)

    batch_seq_size, inputs, targets = to_batches(train_set, batch_size, seq_length, len(model.vocab)+1)
    for epoch in epoch_progress:
        epoch_progress.set_description(f'Epoch {epoch}')
        hc = model.new_hidden(batch_size)
        batch_progress = tqdm.tqdm(range(1, batch_seq_size + 1), position=1)
        running_loss = []
        for (i, inp, target) in zip(batch_progress, inputs, targets):
            if torch.cuda.is_available():
                inp = inp.cuda()
                target = target.cuda()
            hc = list(map(lambda x: torch.autograd.Variable(x.data), hc))
            model.zero_grad()
            out, h = model(inp, hc)
            loss = criterion(out, target.view(-1))
            loss.backward()
            running_loss.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if i > 0 and i % 10 == 0:
                batch_progress.set_description('Batch {} | Loss: {:.4f}'.format(i, loss.item()))
        print(f'Loss Avg: {sum(running_loss) / len(running_loss)}')
        torch.save(model, checkpoint_filename)
    print('elapsed: {:10.4f} seconds'.format(time.time() - start_time))

def test(model: CharRNN, sentence, predict_length=512):
    print("testing")
    with torch.no_grad():
        model.eval()
        # print(f'Begin: {sentence}')
        h = model.new_hidden(1)
        #for ch in sentence:
        c, h = model.predict(sentence, h)
        # result = list(sentence)
        # result.append(c)
        # for _ in range(predict_length):
        # c, h = model.predict(result[-1], h, num_choice=5)
        return c

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--clip', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--pred-len', type=int, default=128)
    parser.add_argument('--corpus-length', type=int, default=256)
    parser.add_argument('--save-checkpoint', type=str, default='char_rnn_checkpoint.pth')
    parser.add_argument('--corpus', type=list,
                        default=['en', 'fr', 'de', 'ja', 'zh-cn', 'zh-tw', 'it', 'ko', 'ru', 'ar', 'hi'])
    args = parser.parse_args()
    if args.train:
        corpus = load_wiki()
        text = ' '.join(corpus)

        char2int, int2char = to_dictionary(text)
        vocab = list(char2int.keys())
        model = CharRNN(vocab, char2int, int2char, n_ts=args.seq_len)
        dataset = np.array([char2int[x] for x in text], dtype=np.longlong)
        train(model, dataset, args.epoch, args.batch_size, args.save_checkpoint,
                grad_clip=args.clip, lr=args.lr, seq_length=args.seq_len)
    else:
        model = torch.load(args.save_checkpoint)
        print(len(model.char2int.keys()))
        print(test(model, 'one giant l', predict_length=args.pred_len))
if __name__ == '__main__':  
    main()
