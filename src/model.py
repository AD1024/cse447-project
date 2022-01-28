import nltk
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from utils import cached

@cached
def load_corpus():
    from nltk.corpus import brown
    vocab = brown.words()
    isword = lambda x: all(map(lambda x: x.isalpha(), list(x)))
    vocab = list(filter(isword, vocab))
    return list(map(lambda x: x.lower(), vocab[:len(vocab) // 20]))

@cached
def load_twitter_corpus():
    from nltk.corpus import twitter_samples
    tokenized = twitter_samples.tokenized()
    isword = lambda x: all(map(lambda x: x.isalpha(), list(x)))
    result = []
    for tokens in tokenized:
        t = list(filter(isword, tokens))
        t = ["<BOS>"] + t
        result += t
    return result

def annotate_sentences(tokens, begin_of_sentence="<BOS>", end_of_sentence="<EOS>"):
    result = [begin_of_sentence]
    for tok in tokens:
        if tok == '.':
            result.append(tok)
            result.append(end_of_sentence)
        else:
            result.append(tok)
    if result[-1] != end_of_sentence:
        result.append(end_of_sentence)
    return result

def to_vec(tokens, begin_of_sentence="<BOS>", end_of_sentence="<EOS>"):
    ident = 3
    mapping = {
        begin_of_sentence: 1,
        end_of_sentence: 2,
    }
    def get_or_add(tok):
        nonlocal ident
        if tok not in mapping:
            mapping[tok] = ident
            ident += 1
        return mapping[tok]
    return list(map(get_or_add, tokens)), mapping

def to_train_data(tokens, sequence_len=5):
    result = []
    tmp = []
    for tok in tokens:
        if len(tmp) == sequence_len:
            result.append(tmp.copy())
            tmp.clear()
        tmp.append(tok)
    while len(tmp) < sequence_len:
        tmp.append("<EOS>")
    result.append(tmp.copy())
    return result

class Model(nn.Module):
    def __init__(self, batch_size, vocab_size, seq_length, num_embedding, num_hidden) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, num_embedding)
        self.lstm = nn.LSTM(num_embedding, num_hidden, num_layers=2, bidirectional=False)
        self.linear_act = nn.Linear(num_hidden * seq_length, vocab_size)
    
    def forward(self, inputs):
        emb = self.embedding(inputs)
        output, hidden = self.lstm(emb)
        output = output.view(output.size(0), -1)
        fc1 = self.linear_act(output)
        return fc1, hidden

def validate(model, inputs, tokenizer):
    inputs = list(map(lambda x: x.lower(), inputs))
    inp = np.asarray(tokenizer.texts_to_sequences([inputs]))
    with torch.no_grad():
        inp = torch.from_numpy(inp)
        pred, _ = model(inp)
        outputs = torch.softmax(pred, dim=1)
        predicted = np.argmax(outputs.numpy())
        print(predicted)
        outputs = outputs.numpy()[0].tolist()
        selections = list(enumerate(outputs))
        selections.sort(key=lambda x: -x[1])
        print(selections[:5])
        print("predicted: ", tokenizer.sequences_to_texts([[x[0] for x in selections[:10]]]))
        print("predicted: ", tokenizer.sequences_to_texts([[predicted]]))

def validate_checkpoint(model="./model.pth"):
    corpus = load_twitter_corpus()
    tokens = annotate_sentences(corpus)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)
    model = torch.load(model)
    validate(model, ["me", "the"], tokenizer)

def train(model, num_epoch=5, batch_size=32, sequence_length=3, lr=0.05, checkpoint_path="./model_twitter.pth", load_model=False):
    print("Loading corpus")
    corpus = load_twitter_corpus()
    tokens = annotate_sentences(corpus)
    tokens = to_train_data(tokens, sequence_len=sequence_length)
    tokenizer = Tokenizer()
    print("Fit on tokenizer")
    tokenizer.fit_on_texts(tokens)
    tokens = np.asarray(tokenizer.texts_to_sequences(tokens))
    tokens = tokens.reshape((-1, sequence_length))
    vocab_size = len(tokenizer.word_counts) + 1
    train_inputs = tokens[:, :-1]
    targets = tokens[: , -1]
    if model is None:
        model = Model(batch_size, vocab_size, (sequence_length - 1), 128, 256)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    inputs = torch.from_numpy(train_inputs)
    targets = np.asarray(to_categorical(targets.tolist(), vocab_size))
    targets = torch.from_numpy(targets.astype("float32"))
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        targets = targets.cuda()
    epoch_progress = tqdm.tqdm(range(num_epoch))
    for epoch in epoch_progress:
        permutation = torch.randperm(inputs.size()[0])
        for i in range(0, inputs.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_inp, batch_target = inputs[indices], targets[indices]
            optimizer.zero_grad()
            pred, hidden = model(batch_inp)
            loss = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()
            if i % (batch_size * 50) == 0:
                epoch_progress.set_description(f"Epoch {epoch} | Loss = {loss}")
        torch.save(model, checkpoint_path)

    validate(model, ["happy", "new"], tokenizer)

if __name__ == '__main__':
    train(None)
    # validate_checkpoint()