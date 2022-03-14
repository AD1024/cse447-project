LANG = ['ar', 'de', 'en', 'fr', 'hi', 'ja', 'ko', 'pt', 'ru', 'sp', 'zh']

aggregated = []

for lang in LANG:
    with open(f'{lang}.json', 'r') as fp:
        content = fp.read()
        if content.count('.') < content.count('。'):
            print(f'{lang} tokenize wrt 。')
            lines = content.split('。')
        else:
            print(f'{lang} tokenize wrt .')
            lines = content.split('.')
        lines = list(filter(lambda x: len(x) > 3, lines))
        lines = lines[:min(15000, len(lines))]
        lines = list(map(lambda x: x.strip().rstrip(), lines))
        lines = ['<bos> ' + ' '.join(list(map(lambda x: '<s>' if x == ' ' else x, x))).strip().rstrip() + ' <eos>\n' for x in lines]
        aggregated += lines

with open('lang-combined.json', 'w') as f:
    f.writelines(aggregated)