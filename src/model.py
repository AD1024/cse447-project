from functools import cache
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from utils import cached

@cached
def load_corpus():
    from nltk.corpus import brown
    vocab = brown.words()
    return list(map(lambda x: x.lower(), vocab[:len(vocab) // 100]))

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

def to_train_data(word_vec, sequence_len=5):
    while len(word_vec) % sequence_len != 0:
        word_vec.append(0)
    return np.asarray(word_vec).reshape((-1, sequence_len))

def to_categorical(word_vec, num_classes):
    one_hot_vecs = np.eye(num_classes)
    return list(map(lambda x: one_hot_vecs[x], word_vec))

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

def validate(model, inputs, mappings):
    inputs = list(map(lambda x: x.lower(), inputs))
    inputs = np.asarray([[mappings.get(x) for x in inputs]])
    lookup = { k : v for (v, k) in mappings.items() }
    with torch.no_grad():
        inp = torch.from_numpy(inputs)
        pred, _ = model(inp)
        outputs = torch.softmax(pred)
        predicted = np.argmax(outputs.numpy())
        word = lookup.get(predicted)
        print("predicted: ", word)

def train(model, num_epoch=100, batch_size=32, sequence_length=3, lr=0.05, checkpoint_path="./model.pth", load_model=False):
    corpus = load_corpus()
    tokens = annotate_sentences(corpus)
    word_vec, word2int = to_vec(tokens)
    vocab_size = len(word2int) + 1
    int2word = {k : v for v, k in word2int.items()}
    word_vec = to_train_data(word_vec, sequence_len=sequence_length)
    train_inputs = word_vec[:, :-1]
    targets = word_vec[: , -1]
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

    validate(model, ["this", "is"])

if __name__ == '__main__':
    train(None)