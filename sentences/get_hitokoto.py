import json
import nltk
import os

texts = []

for x in os.listdir('./'):
    if x.endswith('.json'):
        print(f'processing: {x}')
        with open(x) as fp:
            data = json.load(fp)
            for entry in data:
                content = entry['hitokoto']
                tokenized = ['<bos>'] + list(content) + ['<eos>']
                tokenized_text = ' '.join(tokenized)
                texts.append(tokenized_text + '\n')
    with open('hitokoto.txt', 'w') as fp:
        fp.writelines(texts)