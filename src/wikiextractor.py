import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='text/AA')
    parser.add_argument('--output', required=True)

    args = parser.parse_args()
    with open(args.output, 'w') as out:
        for fname in os.listdir(args.input):
            with open(args.input + '/' + fname) as f:
                line = f.readline()
                while line:
                    j = json.loads(line)
                    if j['text'] != "":
                        out.write(j['text'].replace('\n', ''))
                    line = f.readline()

if __name__ == '__main__':
    main()
