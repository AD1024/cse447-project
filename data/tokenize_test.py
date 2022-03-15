LANG = ['ar', 'de', 'en', 'fr', 'hi', 'ja', 'ko', 'pt', 'ru', 'sp', 'zh']

aggregated = []
answer = []

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
        lines = lines[min(15000, len(lines) - 1000):min(15000, len(lines) - 1000) + 1000]
        lines = list(map(lambda x: x.strip().rstrip(), lines))
        aggregated += list(map(lambda x: x[:-1] + '\n', lines))
        answer += list(map(lambda x: x[-1] + '\n', lines))

with open('combined-test.txt', 'w') as f:
    f.writelines(aggregated)

with open('combined-answer.txt', 'w') as f:
    f.writelines(answer)