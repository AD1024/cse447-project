import torch
import torchtext
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
            res.append(s[:120000])
    return res

def process_quotes(filename):
    text_config = torchtext.legacy.data.Field(init_token='“', eos_token='”')
    dataset = torchtext.legacy.datasets.LanguageModelingDataset(filename, text_config, newline_eos=False)
    text_config.build_vocab(dataset)
    return text_config, dataset

class CharRNN(nn.Module):
    def __init__(self, vocab, text_config, dataset, n_ts=128, embedding_dim=300, hidden_dim=512, num_layers=2, dropout=0.3) -> None:
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_ts = n_ts
        self.vocab = vocab
        self.text_config = text_config
        self.dataset = dataset

        self.embedding = nn.Embedding(len(self.vocab), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            self.num_layers,
                            dropout=self.dropout)
        self.act = nn.Linear(hidden_dim, len(self.vocab))
        self.drop_out = nn.Dropout(self.dropout)
        self.init_weights()

    def init_weights(self):
        rng = 0.1
        self.embedding.weight.data.uniform_(-rng, rng)
        self.act.bias.data.zero_()
        self.act.weight.data.uniform_(-rng, rng)

    def forward(self, x, h):
        out = self.drop_out(self.embedding(x))
        out, (h, c)  = self.lstm(out, h)
        out = self.drop_out(out)
        out = out.contiguous().view(out.size(0) * out.size(1), self.hidden_dim)
        act = self.act(out)
        return act, (h, c)

    def predict(self, characters, h=None, num_choice=3):
        if h is None:
            h = self.new_hidden(1)
        inp = np.array([[self.char2int[x] if x in self.char2int else 0 for x in characters]], dtype=np.longlong)
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
        weights = next(self.parameters())
        h, c = weights.new_zeros((self.num_layers, batch_size, self.hidden_dim)),\
               weights.new_zeros((self.num_layers, batch_size, self.hidden_dim))
        h.zero_()
        c.zero_()
        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()
        return h, c

def detach_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(detach_hidden(v) for v in h)

def generate_quote(model: CharRNN, text_config, sample_size=3):
    model.eval()
    idx = text_config.vocab.stoi["“"]
    x = torch.Tensor([idx]).view(1, 1).long().to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    poem = []
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for _ in range(128):
            output, hidden = model(x, hidden)
            output = output.view(model.vocab_size)
            if sample_size > 1:
                probs = F.softmax(output, dim=0).cpu().numpy()
                probs /= probs.sum()
                idx = np.random.choice(range(model.vocab_size), p=probs)
            else:
                idx = torch.argmax(output)
            if idx == text_config.vocab.stoi["”"]: break
            poem.append(text_config.vocab.itos[idx])
            x = torch.Tensor([idx]).view(1, 1).long().to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    return ''.join(poem)

def train_quotes(model: CharRNN, dataset, num_epoch, batch_size, checkpoint_filename, seq_length=128, grad_clip=1, lr=0.001):
    if torch.cuda.is_available():
        model = model.cuda()
    train_iter = torchtext.legacy.data.BPTTIterator(dataset,
        batch_size,
        seq_length,
        repeat=False,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    vocab_size = len(dataset.fields['text'].vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = []
    for epoch in range(num_epoch):
        try:
            model.train()
            epoch_loss = []
            train_iter.init_epoch()
            progress = tqdm.tqdm(train_iter)
            for i, batch in enumerate(progress):
                if i % 10 == 0:
                    progress.set_description(f"Batch #{i}")
                # if hidden is None:
                hidden = model.new_hidden(batch.batch_size)
                # else:
                #     hidden = detach_hidden(hidden)

                text, target = batch.text, batch.target
                output, _ = model(text, hidden)
                optimizer.zero_grad()
                loss = criterion(output.view(-1, vocab_size), target.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            epoch_loss = np.mean(epoch_loss)
            print(f'Epoch {epoch}: loss = {epoch_loss}')
        except KeyboardInterrupt:
            torch.save(model.state_dict(), "{}_{0:d}.pth".format(checkpoint_filename, epoch))
            return total_loss
    torch.save(model, f'{checkpoint_filename}')
    return total_loss

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
        # corpus = load_corpus()
        # text = ' '.join(corpus)
        # corpus = []
        # for lang in args.corpus:
        #     corpus += load_wikitext(lang=lang, num_samples=args.corpus_length)
        # corpus = load_wiki()
        # text = ' '.join(corpus)

        # char2int, int2char = to_dictionary(text)
        # vocab = list(char2int.keys())
        # model = CharRNN(vocab, char2int, int2char)
        # dataset = np.array([char2int[x] for x in text], dtype=np.longlong)
        text_config, dataset = process_quotes('data/quotes.txt')
        model = CharRNN(dataset.fields['text'].vocab, text_config, dataset)
        train_quotes(model, dataset, args.epoch, args.batch_size, args.save_checkpoint,
                grad_clip=args.clip, lr=args.lr, seq_length=args.seq_len)
    else:
        model: CharRNN = torch.load(args.save_checkpoint)
        print(generate_quote(model, model.text_config))
        # print(len(model.char2int.keys()))
        # print(test(model, '中国邮', predict_length=args.pred_len))
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
if __name__ == '__main__':  
    main()
