import datasets
import nltk

data = datasets.load_dataset('Abirate/english_quotes')
quote = []
for row in data['train']:
    text = row['quote']
    text = text.replace('“', '“ ')
    text = text.replace('”', '” ')
    text = nltk.word_tokenize(text)
    text = ' '.join(text)
    text += '\n'
    quote.append(text)

with open('quotes.txt', 'w') as fp:
    fp.writelines(quote)