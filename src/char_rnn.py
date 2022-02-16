import torch
import torch.nn as nn
import tqdm
import argparse
import numpy as np
from torch.nn import functional as F
from datasets import load_dataset
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

def load_wikitext(lang='zh-cn', num_samples=1024):
    annotations = ['_START_ARTICLE_', '_START_SECTION_', '_START_PARAGRAPH_', '\n', '_NEWLINE_']
    dataset = load_dataset('wiki40b', lang, split='train', beam_runner='DirectRunner')
    result = []
    data = dataset['text']
    for i in range(num_samples):
        text = data[i]
        for x in annotations:
            text = text.replace(x, '')
        result.append(text)
    return result

@cached
def load_wiki():
    res = []
    progress = tqdm.tqdm(list(os.listdir('wiki')))
    for fname in progress:
        progress.set_description('loading wiki/' + fname)
        with open('wiki/' + fname) as f:
            i = 1000
            line = f.readline()
            while line and i > 0:
                line = line.split('\t')
                # source lang
                res.append(line[1])
                # target lang
                res.append(line[2][:-1])
                line = f.readline()
                i -= 1
    random.shuffle(res)
    return res

class CharRNN(nn.Module):
    def __init__(self, vocab, char2int, int2char, n_ts=128, hidden_dim=512, num_layers=2, dropout=0.5) -> None:
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_ts = n_ts
        self.vocab = vocab
        self.char2int = char2int
        self.int2char = int2char

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
        if torch.cuda.is_available():
            inp = inp.cuda()
        out, h = self.forward(inp, h)
        prob = F.softmax(out[-1], dim=0)
        prob, ch = prob.topk(num_choice)
        prob = prob.cpu().numpy()
        ch = ch.cpu().numpy()
        ans = np.random.choice(ch, p=prob / prob.sum())
        return self.int2char[ans], h
    
    def new_hidden(self, batch_size):
        h, c = torch.autograd.Variable(torch.randn(self.num_layers, batch_size, self.hidden_dim)),\
               torch.autograd.Variable(torch.randn(self.num_layers, batch_size, self.hidden_dim))
        h.zero_()
        c.zero_()
        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()
        return h, c

def to_batches(data, num_seq, seq_length, volcab_len):
    size_per_batch = num_seq * seq_length
    batch_size = len(data) // size_per_batch
    data = data[:batch_size * size_per_batch].reshape((num_seq, -1))
    size = data.shape[1] // seq_length
    inputs = []
    targets = []
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
        inputs.append(one_hot_vector(inp, volcab_len))
        targets.append(target)
    return size, \
        torch.autograd.Variable(torch.from_numpy(np.concatenate(inputs).reshape(batch_size, -1, seq_length, volcab_len))), \
        torch.autograd.Variable(torch.from_numpy(np.array(targets).reshape(batch_size, -1, seq_length)))

def train(model: CharRNN, data, num_epoch, batch_size, seq_length=128, grad_clip=1, lr=0.001):
    start_time = time.time()
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_set = data[:int(0.9 * len(data))]
    epoch_progress = tqdm.tqdm(range(num_epoch), position=0)

    batch_seq_size, inputs, targets = to_batches(train_set, batch_size, seq_length, len(model.vocab))
    for epoch in epoch_progress:
        epoch_progress.set_description(f'Epoch {epoch}')
        hc = model.new_hidden(batch_size)
        batch_progress = tqdm.tqdm(range(1, batch_seq_size + 1), position=1)
        for (i, inp, target) in zip(batch_progress, inputs, targets):
            if torch.cuda.is_available():
                inp = inp.cuda()
                target = target.cuda()
            hc = list(map(lambda x: torch.autograd.Variable(x.data), hc))
            model.zero_grad()
            out, h = model(inp, hc)
            loss = criterion(out, target.view(batch_size * seq_length))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if i > 0 and i % 10 == 0:
                batch_progress.set_description('Batch {} | Loss: {:.4f}'.format(i, loss.item()))
        torch.save(model, 'char_rnn_comments.pth')
    print('elapsed: {:10.4f} seconds'.format(time.time() - start_time))

def test(model: CharRNN, sentence, predict_length=512):
    print("testing")
    with torch.no_grad():
        model.eval()
        sentence = sentence.lower()
        # print(f'Begin: {sentence}')
        h = model.new_hidden(1)
        for ch in sentence:
            c, h = model.predict(ch, h)
        result = list(sentence)
        result.append(c)
        for _ in range(predict_length):
            c, h = model.predict(result[-1], h, num_choice=5)
            result.append(c)
        return ''.join(result)

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
    parser.add_argument('--corpus', type=list,
                        default=['en', 'fr', 'de', 'ja', 'zh-cn', 'zh-tw', 'it', 'ko', 'ru', 'ar', 'hi'])
    args = parser.parse_args()
    if args.train:
        # corpus = load_corpus()
        # text = ' '.join(corpus)
        # corpus = []
        # for lang in args.corpus:
        #     corpus += load_wikitext(lang=lang, num_samples=args.corpus_length)
        corpus = load_comments()
        text = ' '.join(corpus)

        char2int, int2char = to_dictionary(text)
        vocab = list(char2int.keys())
        model = CharRNN(vocab, char2int, int2char)
        dataset = np.array([char2int[x] for x in text])
        train(model, dataset, args.epoch, args.batch_size,
                grad_clip=args.clip, lr=args.lr, seq_length=args.seq_len)
    else:
        model = torch.load('char_rnn_comments.pth')
        print(len(model.char2int.keys()))
        print(test(model, '嘉人们', predict_length=args.pred_len))
        # print(text[-2])
        #     model = torch.load('char_rnn.pth')
        #     inputs = '''Happ
        # Happy Ne
        # Happy New Yea
        # That’s one small ste
        # That’s one sm
        # That’
        # Th
        # one giant leap for mankin
        # one giant leap fo
        # one giant lea
        # one giant l
        # one gia
        # on'''.split('\n')
        #     for input in inputs:
        #         print(input)
        #         print(test(model, input, predict_length=1))

main()
