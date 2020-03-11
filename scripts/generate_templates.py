import json
import argparse

# "query" is the attribute that is queried
QUERY_PH = '%query%'
Q_PH = '%Q%'
# "attr" is the attribute that is used for reasoning
ATTR_PH = '%attr%'
A_PH = '%A%'

VALUES = [('color', 'C'), ('shape', 'S'),
          ('material', 'M'), ('size', 'Z')]

parser = argparse.ArgumentParser('Generate templates from a meta-template')
parser.add_argument('meta_template')
parser.add_argument('templates')
parser.add_argument('--query-material', type=int, default=1)
args = parser.parse_args()

with open(args.meta_template) as src:
    meta_template = src.read()

str_templates = set()
for query in VALUES if args.query_material else [t for t in VALUES if t[0] != 'material']:
    for attr in VALUES:
        template = meta_template
        if query[0] == attr[0]:
            continue
        template = template.replace(QUERY_PH, query[0])
        template = template.replace(Q_PH, query[1])
        template = template.replace(ATTR_PH, attr[0])
        template = template.replace(A_PH, attr[1])
        str_templates.add(template)

templates = []
for str_ in str_templates:
    templates.append(json.loads(str_)[0])

with open(args.templates, 'w') as dst:
    json.dump(templates, dst, indent=2)
