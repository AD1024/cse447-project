from wikitextprocessor import Wtp, WikiNode, NodeKind
import argparse
import tqdm

def process_children(root: WikiNode):
    result = ''
    for ch in root.children:
        if isinstance(ch, str):
            result += ch.replace('\n', '')
        elif isinstance(ch, WikiNode):
            if ch.kind in (NodeKind.LINK, NodeKind.TABLE, NodeKind.LIST,
                            NodeKind.LIST_ITEM, NodeKind.TABLE_CAPTION,
                            NodeKind.TABLE_CELL, NodeKind.TABLE_HEADER_CELL,
                            NodeKind.TABLE_ROW):
                continue
            else:
                result += process_children(ch)
    return result

def parse_page(model, title, text):
    ctx = Wtp()
    if model == 'wikitext' and not title.startswith('Template:'):
        ctx.analyze_templates()
        ctx.start_page(title)
        root = ctx.parse(text)
        result = process_children(root)
        return title, result
    else:
        return None, None

def parse_file(filename, output_file, page_limit, num_threads=4):
    ctx = Wtp(num_threads=num_threads, quiet=True)
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