#!/usr/bin/env python
import os
from statistics import mode
import string
import random
import torch
from char_rnn import CharRNN, load_wiki, test as predict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self, pt_model):
        self.model = pt_model

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        return []

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f.readlines():
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        pass

    def run_pred(self, data):
        # your code here
        preds = []
        # all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            # predict language of the word
            inp = inp.lower()
            sentence = inp
            preds.append(''.join(map(lambda x: ' ' if x == '<s>' else '' if x == '<eos>' else x, predict(self.model, sentence))))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
        #     f.write('dummy save')
        pass

    @classmethod
    def load(cls, work_dir):
        # your code here
        text_config, dataset = load_wiki(os.path.join(work_dir, 'lang-combined.json'))
        model = CharRNN(dataset.fields['text'].vocab, text_config, n_ts=64)
        model.load_state_dict(torch.load(os.path.join(work_dir, '447_char_rnn_checkpoint_4.pth'), map_location='cpu'))
        if torch.cuda.is_available():
            model.to(torch.device('cuda'))
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    # random.seed(0)
    if not os.path.isdir(args.work_dir):
        os.makedirs(args.work_dir, exist_ok=True)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel.load(args.work_dir)
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel(MyModel.load(args.work_dir))
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
