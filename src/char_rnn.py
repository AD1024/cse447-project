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

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@cached
def load_wiki(filename='data/lang-combined.json'):
    print('Loading dataset')
    data_config = torchtext.data.Field(init_token='<bos>', eos_token='<eos>')
    dataset = torchtext.datasets.LanguageModelingDataset(f'{filename}', data_config, newline_eos=False)
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
        inp = torch.from_numpy(np.array([[self.text_config.vocab[x] for x in characters]], dtype=np.longlong)).transpose(0, 1)
        # inp = torch.autograd.Variable(torch.from_numpy(one_hot_vector(inp, len(self.vocab)+1)))
        if torch.cuda.is_available():
            inp = inp.to(DEVICE)
        out, h = self.forward(inp, h)
        out = out.view(-1, len(self.vocab))
        prob = F.softmax(out[-1], dim=0)
        prob, ch = prob.topk(num_choice)
        prob = prob.cpu().numpy()
        ch = ch.cpu().numpy()
        return [self.text_config.vocab.itos[x] for x in ch], h
    
    def new_hidden(self, batch_size):
        w = next(self.parameters())
        h, c = w.new_zeros((self.num_layers, batch_size, self.hidden_dim)),\
               w.new_zeros((self.num_layers, batch_size, self.hidden_dim))
        if torch.cuda.is_available():
            h = h.to(DEVICE)
            c = c.to(DEVICE)
        return h, c

def detach_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(detach_hidden(v) for v in h)

def train(model: CharRNN, dataset, num_epoch, batch_size, checkpoint_filename, seq_length=128, grad_clip=1, lr=0.001):
    start_time = time.time()
    if torch.cuda.is_available():
        model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_progress = tqdm.tqdm(range(num_epoch), position=0)
    total_loss = []
    hidden = None
    print(f'Train with batch size = {batch_size}')
    train_iter = torchtext.data.BPTTIterator(dataset,
        batch_size,
        seq_length,
        shuffle=True,
        repeat=False,
        device=DEVICE)
    vocab_size = len(dataset.fields['text'].vocab)
    for epoch in range(num_epoch):
        try:
            model.train()
            epoch_loss = []
            train_iter.init_epoch()
            progress = tqdm.tqdm(train_iter)
            for i, batch in enumerate(progress):
                if i % 10 == 0:
                    progress.set_description(f"Batch #{i} | RL = {np.mean(epoch_loss)}")
                # if hidden is None:
                hidden = model.new_hidden(batch.batch_size)
                model.zero_grad()
                loss = 0
                # else:
                #     hidden = detach_hidden(hidden)
                text, target = batch.text, batch.target
                for i in range(text.size(0)):
                    output, hidden = model(text[i, :].unsqueeze(0), hidden)
                    loss += criterion(output.view(-1, vocab_size), target[i, :].view(-1))
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_loss.append(loss.item() / text.size(0))

            epoch_loss = np.mean(epoch_loss)
            print(f'Epoch {epoch}: loss = {epoch_loss}')
        except KeyboardInterrupt:
            torch.save(model.state_dict(), "{}.pth".format(checkpoint_filename))
            return total_loss
        torch.save(model.state_dict(), checkpoint_filename)
    torch.save(model.state_dict(), f'{checkpoint_filename}')
    print('elapsed: {:10.4f} seconds'.format(time.time() - start_time))
    return total_loss

def test(model: CharRNN, sentence, predict_length=512):
    sentence = list(map(lambda x: '<s>' if x == ' ' else x, sentence))
    sentence = ['<bos>'] + sentence
    model.to(DEVICE)
    with torch.no_grad():
        model.eval()
        # print(f'Begin: {sentence}')
        h = model.new_hidden(1)
        result = []
        for ch in sentence:
            c, h = model.predict(ch, h)
            result.append(c)
        # result = list(sentence)
        # result.append(c)
        # for _ in range(predict_length):
        # c, h = model.predict(result[-1], h, num_choice=5)
        return result[-1]

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
        text_config, dataset = load_wiki()
        model = CharRNN(dataset.fields['text'].vocab, text_config, n_ts=args.seq_len)
        model.load_state_dict(torch.load(args.save_checkpoint))
        # print(test(model, 'さような', predict_length=args.pred_len))
        with open('data/combined-test.txt') as fp:
            lines = fp.readlines()
            lines = [x.replace('\n', '') for x in lines]
            correct = 0
            with open('data/combined-answer.txt') as ans:
                ans = ans.readlines()
                ans = [x.replace('\n', '') for x in ans]
                for inp, target in zip(lines, ans):
                    answer = test(model, inp)
                    print(answer, target, target in answer)
                    if target in answer:
                        correct += 1
                print(f'Accuracy: {correct / len(lines)}')

if __name__ == '__main__':  
    main()
