from sklearn.utils import shuffle
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
def load_wiki(filename='lang-combined.json'):
    data_config = torchtext.legacy.data.Field(init_token='<bos>', eos_token='<eos>')
    dataset = torchtext.legacy.datasets.LanguageModelingDataset(f'data/{filename}', data_config, newline_eos=False)
    data_config.build_vocab(dataset)
    return data_config, dataset

class CharRNN(nn.Module):
    def __init__(self, vocab, text_config, n_ts=128, embedding_dim=300, hidden_dim=512, num_layers=3, dropout=0.3) -> None:
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_ts = n_ts
        self.vocab = vocab
        self.text_config = text_config

        self.drop_out = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(len(self.vocab), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            self.num_layers,
                            dropout=self.dropout)
        self.act = nn.Linear(hidden_dim, len(self.vocab))

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

def detach_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(detach_hidden(v) for v in h)

def train(model: CharRNN, dataset, num_epoch, batch_size, checkpoint_filename, seq_length=128, grad_clip=1, lr=0.001):
    start_time = time.time()
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_progress = tqdm.tqdm(range(num_epoch), position=0)
    total_loss = []
    hidden = None
    train_iter = torchtext.legacy.data.BPTTIterator(dataset,
        batch_size,
        seq_length,
        shuffle=True,
        repeat=False,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    vocab_size = len(dataset.fields['text'].vocab)
    for epoch in range(num_epoch):
        try:
            model.train()
            epoch_loss = []
            train_iter.init_epoch()
            progress = tqdm.tqdm(train_iter)
            for i, batch in enumerate(progress):
                if i % 10 == 0:
                    progress.set_description(f"Batch #{i}")
                if hidden is None:
                    hidden = model.new_hidden(batch.batch_size)
                else:
                    hidden = detach_hidden(hidden)

                text, target = batch.text, batch.target
                output, hidden = model(text, hidden)
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
    torch.save(model.state_dict(), f'{checkpoint_filename}')
    print('elapsed: {:10.4f} seconds'.format(time.time() - start_time))
    return total_loss

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
        text_config, dataset = load_wiki()
        model = CharRNN(dataset.fields['text'].vocab, text_config, n_ts=args.seq_len)
        train(model, dataset, args.epoch, args.batch_size, args.save_checkpoint,
                grad_clip=args.clip, lr=args.lr, seq_length=args.seq_len)
    else:
        model = torch.load(args.save_checkpoint)
        print(len(model.char2int.keys()))
        print(test(model, 'one giant l', predict_length=args.pred_len))
if __name__ == '__main__':  
    main()
