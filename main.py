import argparse
import json
from keyword import iskeyword

import sympy

from symbolic import Scope, FeatureModel, generateAnnotations, Expression, SymbolWeights, factories
from lists import VariationalElement, generateVariationalList


# Prevents variables from messing with the parsing of feature models and annotations when using eval.
def checkVars(vars):
    for var in vars:
        if not var.isidentifier or iskeyword(var):
            print("Invalid variable name:", var)
            exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['annotations', 'featuremodel', 'domainmodel', 'dm', 'fm', 'an'])
    parser.add_argument('input', type=argparse.FileType('r'))
    parser.add_argument('output', type=str)
    args = parser.parse_args()
    config = json.loads(args.input.read())

    if args.command in  ('annotations', 'an'):
        checkVars(config['vars'])
        scope = Scope(list(sympy.symbols(' '.join(config['vars']))))
        fm = FeatureModel(Expression.deserialize(config['fm'], scope))
        print("Loaded feature model and scope")
        annotations = generateAnnotations(scope, fm, config['n'], config['min'], config['max'], config['mean'])
        print("Generated")
        config["annotations"] = list(map(lambda x: x.serialize(), annotations))
        out = open(args.output, 'w')
        out.write(json.dumps(config))
        out.close()
    elif args.command in ('featuremodel', 'fm'):
        checkVars(config['vars'])
        scope = Scope(list(sympy.symbols(' '.join(config['vars']))))
        fm = FeatureModel(factories[2](scope).newExpression(SymbolWeights(10, scope)))
        config['fm'] = fm.expr.serialize()
        out = open(args.output, 'w')
        out.write(json.dumps(config))
        out.close()
    elif args.command in ('domainmodel', 'dm'):
        checkVars(config['vars'])
        scope = Scope(list(sympy.symbols(' '.join(config['vars']))))
        annotations = list(map(lambda x: Expression.deserialize(x, scope), config["annotations"]))
        size = config['size']
        elementRepetitions = config['elementRepetitions']
        contiguousSublists = config['contiguousSublists']
        res = generateVariationalList(size, set(range(size)), elementRepetitions, contiguousSublists, annotations)
        config['model'] = list(map(VariationalElement.encode, res))
        out = open(args.output, 'w')
        out.write(json.dumps(config))
        out.close()