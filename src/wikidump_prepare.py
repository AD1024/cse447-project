from wikitextprocessor import Wtp, WikiNode, NodeKind
import argparse
import tqdm

def process_children(root: WikiNode):
    result = ''
    for ch in root.children:
        if isinstance(ch, str):
            result += ch.replace('\n', '')
        elif isinstance(ch, WikiNode):
            if ch.kind in (NodeKind.TABLE, NodeKind.TABLE_CAPTION,
                            NodeKind.TABLE_CELL, NodeKind.TABLE_HEADER_CELL,
                            NodeKind.TABLE_ROW):
                continue
            else:
                if ch.kind == NodeKind.LINK:
                    ch_result = None
                    if len(ch.args) > 0:
                        if len(ch.args[0]) > 0:
                            ch_result = ch.args[0][0]
                            if isinstance(ch_result, str) and (ch_result.startswith('File:') or ch_result.startswith('Category:')):
                                result += process_children(ch)
                                continue
                    if len(ch.args) > 1:
                        if len(ch.args[1]) > 0:
                            ch_result = ch.args[1][0]
                    if ch_result is not None:
                        if isinstance(ch_result, str):
                            result += ch_result
                        elif isinstance(ch, WikiNode):
                            result += process_children(ch_result)
                            result += ' '
                result += process_children(ch)
    return result\
            .replace("()", '')\
            .replace("[]", '')\
            .replace('\{\}', '')\
            .replace('（）', '()')\
            .replace('-{', '')\
            .replace('-}', '')\
            .replace('　', '')\
            .replace('  ', ' ')

def parse_page(model, title, text):
    ctx = Wtp(quiet=True)
    if model == 'wikitext' and not title.startswith('Template:') and not title.startswith('Wikipedia:'):
        ctx.analyze_templates()
        ctx.start_page(title)
        try:
            root = ctx.parse(text)
            result = process_children(root)
            return title, result.replace('  ', ' ')
        except Exception as e:
            if isinstance(e, InterruptedError):
                exit(1)
            return None, None
    else:
        return None, None

def parse_file(filename, output_file, page_limit, num_threads=4):
    ctx = Wtp(num_threads=num_threads)
    result = ctx.process(filename, parse_page)
    article = []
    for ((title, parsed_body), _) in zip(result, range(page_limit)):
        if title is None:
            continue
        else:
            article.append(parsed_body)
    with open(output_file, 'w') as fp:
        fp.writelines(article)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--page-limit', type=int, required=True)

    args = parser.parse_args()
    parse_file(args.filename, args.output, args.page_limit)

if __name__ == '__main__':
    main()
