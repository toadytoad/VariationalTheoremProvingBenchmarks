import itertools
import random
import string
import zlib
from enum import Enum
from functools import reduce
from typing import Callable, List, Tuple, Dict

import pyapproxmc
import sympy
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from scipy.stats import gaussian_kde
from bitarray import bitarray


class Scope:
    def __init__(self, v: List[sympy.Symbol]):
        self.variables = v
        self.inverse = {v[i]: i for i in range(len(v))}
        self.truthTables = []
        for i in range(len(v)):
            tt = bitarray(1 << len(v))
            for j in range(1 << len(v)):
                if j & (1 << i):
                    tt[j] = 1
            self.truthTables.append(tt)

    def __len__(self):
        return len(self.variables)

    def __repr__(self):
        return "Symbolic Scope with the following variables:\n" + str(self.variables)


class Substitution:
    def __init__(self, sub: Dict[sympy.Symbol, bool], scope: Scope):
        for i in sub:
            assert i in scope.variables
        assert len(sub) == len(scope.variables)
        self.scope = scope
        self.sub = sub

    def __repr__(self):
        return "Substitution:\n" + str(self.sub)


class Operation(Enum):
    SYMBOL = 0
    NOT = 1
    AND = 2
    OR = 3
    EQUALS = 4


operationStrs = [None, "~", "&", "|", "=="]
operationFuncs = [None, lambda x: not x, lambda *x: reduce(lambda i, j: i & j, x, True),
                  lambda *x: reduce(lambda i, j: i | j, x, False), lambda x, y: x == y]
truthTableFuncs = [None, lambda x: ~x, lambda x, size: reduce(lambda i, j: i & j, x, ~bitarray(size)),
                   lambda x, size: reduce(lambda i, j: i | j, x, bitarray(size)), lambda x, y: ~(x ^ y)]


class Expression:
    def __init__(self, op, children: List[sympy.Symbol or "Expression"], scope: Scope):
        if op == Operation.SYMBOL:
            assert len(children) == 1
            assert type(children[0]) is sympy.Symbol
        elif op == Operation.NOT:
            assert len(children) == 1
        elif op == Operation.EQUALS:
            assert len(children) == 2
        self.op = op
        self.children = children
        self.scope = scope

    def __repr__(self):
        if self.op == Operation.SYMBOL:
            return str(self.children[0])
        elif self.op == Operation.NOT:
            return "~" + str(self.children[0]) + ""
        elif self.op == Operation.EQUALS:
            return "~(" + str(self.children[0]) + "^" + str(self.children[1]) + ")"
        elif self.op == Operation.AND:
            return operationStrs[self.op.value].join(map(str, self.children))
        else:
            return "(" + operationStrs[self.op.value].join(map(str, self.children)) + ")"

    def applySubstitution(self, substitution: Substitution):
        if self.op == Operation.SYMBOL:
            return substitution.sub[self.children[0]]
        return operationFuncs[self.op.value](*map(lambda x: x.applySubstitution(substitution), self.children))
        # kinda unsafe in terms of arguments, but it should
        # be fine because of the assertions in __init__

    def getTruthTable(self):
        if self.op == Operation.SYMBOL:
            return self.scope.truthTables[self.scope.inverse[self.children[0]]]
        if self.op == Operation.NOT:
            return ~self.children[0].getTruthTable()
        if self.op == Operation.EQUALS:
            return truthTableFuncs[self.op.value](*map(lambda x: x.getTruthTable(), self.children))
        return truthTableFuncs[self.op.value](map(lambda x: x.getTruthTable(), self.children),
                                              1 << len(self.scope.variables))

    def __str__(self):
        return repr(self)

    def __or__(self, other: "Expression"):
        assert self.scope == other.scope
        return Expression(Operation.OR, [self, other], self.scope)

    def __and__(self, other: "Expression"):
        assert self.scope == other.scope
        return Expression(Operation.AND, [self, other], self.scope)

    def __invert__(self):
        return Expression(Operation.NOT, [self], self.scope)

    def __eq__(self, other: "Expression"):
        assert self.scope == other.scope
        return Expression(Operation.EQUALS, [self, other], self.scope)

    def toCNF(self):
        if self.op == Operation.SYMBOL:
            return CNF([CNFTerm([Atom(self.children[0], False, self.scope)], self.scope)], self.scope)
        elif self.op == Operation.NOT:
            return ~self.children[0].toCNF()
        elif self.op == Operation.AND:
            start = self.children[0].toCNF()
            for i in range(1, len(self.children)):
                start &= self.children[i].toCNF()
            return start
        elif self.op == Operation.OR:
            start = self.children[0].toCNF()
            for i in range(1, len(self.children)):
                start |= self.children[i].toCNF()
            return start
        else:
            return NotImplementedError("Only Or, And, and Not are supported at this time.")


# TODO to cnf form, create a cnf type that can be anded or ored with another cnf type to form another
# For and, it's easy, just combine the terms together
# For Or, distribute each term. CNF: List[CNFTerm], CNFTerm: List[Atom]

class Atom:
    def __init__(self, symbol: sympy.Symbol, negation: bool, scope: Scope):
        assert symbol in scope.variables
        self.symbol = symbol
        self.negation = negation
        self.scope = scope

    def __eq__(self, other: "Atom"):
        return self.symbol == other.symbol and self.negation == other.negation and self.scope is other.scope

    def __invert__(self):
        return Atom(self.symbol, not self.negation, self.scope)

    def __repr__(self):
        if self.negation:
            return '~' + str(self.symbol)
        else:
            return str(self.symbol)


class CNFTerm:
    def __init__(self, atoms: List[Atom], scope: Scope):
        self.atoms = atoms
        assert all(atoms[i].scope is scope for i in range(len(atoms)))
        self.scope = scope

    def __or__(self, other: "CNFTerm"):
        assert self.scope is other.scope
        return CNFTerm(self.atoms + other.atoms, self.scope).simplify()

    def simplify(self):
        i = 0
        while i < len(self.atoms) - 1:
            j = i + 1
            while j < len(self.atoms):
                if self.atoms[i].symbol == self.atoms[j].symbol:
                    if self.atoms[i].negation ^ self.atoms[j].negation:
                        self.atoms = []
                        return self
                    else:
                        self.atoms.pop(j)
                        j -= 1
                j += 1
            i += 1
        return self

    def __le__(self, other: "CNFTerm"):
        return all(i in other.atoms for i in self.atoms)

    def __repr__(self):
        return '(' + '|'.join(map(str, self.atoms)) + ')'

    def __invert__(self):
        return CNF([CNFTerm([~i], self.scope) for i in self.atoms], self.scope)

    def __str__(self):
        return self.__repr__()

    def toClause(self):
        return [(-(self.scope.inverse[i.symbol] + 1)) if i.negation else self.scope.inverse[i.symbol] + 1 for i in
                self.atoms]


class CNF:
    def __init__(self, terms: List[CNFTerm], scope: Scope):
        self.terms = terms
        assert all(terms[i].scope is scope for i in range(len(terms)))
        self.scope = scope

    def __and__(self, other: "CNF"):
        assert other.scope is self.scope
        return CNF(self.terms + other.terms, self.scope)

    def __or__(self, other: "CNF"):
        assert other.scope is self.scope
        other.simplify()
        self.simplify()
        return CNF([i | j for i, j in itertools.product(self.terms, other.terms)], self.scope).simplify()

    def __invert__(self):
        start = ~self.terms[0]
        for i in range(1, len(self.terms)):
            start |= ~self.terms[i]
        return start

    def simplify(self):
        for t in self.terms:
            t.simplify()
        self.terms = [t for t in self.terms if len(t.atoms)]
        i = 0
        while i < len(self.terms) - 1:
            j = i + 1
            while j < len(self.terms):
                if self.terms[i] <= self.terms[j]:
                    self.terms.pop(j)
                    j -= 1
                elif self.terms[j] <= self.terms[i]:
                    self.terms.pop(i)
                    i -= 1
                    break
                j += 1
            i += 1
        return self

    def __repr__(self):
        return '&'.join(map(str, self.terms))

    def __str__(self):
        return self.__repr__()

    def getWeight(self):
        clauses = [i.toClause() for i in self.terms]
        mx = 0
        for i in clauses:
            mx = max(mx, max(map(abs, i)))
        c = pyapproxmc.Counter()
        c.add_clauses(clauses)
        count = c.count()
        weight = count[0] * 2 ** (count[1]+len(self.scope)-mx)
        return weight


class WeightsCollection:
    def __init__(self, weights: List[Tuple[float, object]]):
        self.weights = weights
        self.sum = sum(weights[i][0] for i in range(len(weights)))
        self.psa = [weights[0][0]]
        for i in range(1, len(self.weights)):
            self.psa.append(self.psa[-1] + self.weights[i][0])

    def __getitem__(self, rand: random.Random) -> object:
        val = rand.random() * self.sum
        for i in range(len(self.weights)):
            if val < self.psa[i]:
                return self.weights[i][1]


class VariableWeights:
    def __init__(self, weights: List[Tuple[Callable[[int], float], Operation]]):
        self.cache = {}
        self.weights = weights

    def __getitem__(self, index) -> WeightsCollection:
        if index not in self.cache:
            weight = [(self.weights[i][0](index), self.weights[i][1]) for i in range(len(self.weights))]
            self.cache[index] = WeightsCollection(weight)
        return self.cache[index]


class SymbolWeights:
    def __init__(self, factor: float, scope: Scope):
        self.scope = scope
        self.weights = [1] * len(self.scope)
        self.f = factor

    def __getitem__(self, rand: random.Random):
        psa = [self.weights[0]]
        for i in range(1, len(self.weights)):
            psa.append(self.weights[i] + psa[-1])
        val = rand.random() * psa[-1]
        for i in range(len(self.weights)):
            if val < psa[i]:
                self.weights[i] /= self.f
                return self.scope.variables[i]


class RandomExpressionFactory:
    def __init__(self, weights: VariableWeights, rand: random.Random, scope: Scope, andWeights: WeightsCollection,
                 orWeights: WeightsCollection):
        self.weights = weights
        self.rand = rand
        self.scope = scope
        self.andWeights = andWeights
        self.orWeights = orWeights

    def newExpression(self, sw: SymbolWeights, depth=0) -> Expression:
        weight = self.weights[depth]
        op = weight[self.rand]
        if op == Operation.SYMBOL:
            negate = self.rand.getrandbits(1)
            if negate:
                return ~Expression(op, [sw[self.rand]], self.scope)
            else:
                return Expression(op, [sw[self.rand]], self.scope)

        elif op == Operation.NOT:
            return Expression(op, [self.newExpression(sw, depth + 1)], self.scope)
        elif op == Operation.EQUALS:
            return Expression(op, [self.newExpression(sw, depth + 1), self.newExpression(sw, depth + 1)], self.scope)
        elif op == Operation.AND:
            return Expression(op, [self.newExpression(sw, depth + 1) for z in range(self.andWeights[self.rand])],
                              self.scope)
        elif op == Operation.OR:
            return Expression(op, [self.newExpression(sw, depth + 1) for z in range(self.orWeights[self.rand])],
                              self.scope)


from lists import *

factories = [lambda scope: RandomExpressionFactory(VariableWeights([(lambda x: (3) * x, Operation.SYMBOL),
                                                                    (lambda x: (0 if x % 2 else 3), Operation.AND),
                                                                    (lambda x: (3 if x % 2 else 0), Operation.OR)]),
                                                   rand, scope, WeightsCollection([(1, 2), (7, 3), (3, 4)]),
                                                   WeightsCollection([(9, 2), (0.8, 3), (0.2, 4)])),
             lambda scope: RandomExpressionFactory(VariableWeights([(lambda x: (x - 1) * x, Operation.SYMBOL),
                                                                    (lambda x: (0 if x % 2 else 3), Operation.AND),
                                                                    (lambda x: (2 if x % 2 else 0), Operation.OR)]),
                                                   rand, scope, WeightsCollection([(10, 2), (3, 3)]),
                                                   WeightsCollection([(10, 2)])),
             lambda scope: RandomExpressionFactory(VariableWeights([(lambda x: 1.5 * (x - 1) * x, Operation.SYMBOL),
                                                                    (lambda x: (0 if x % 2 else 2), Operation.AND),
                                                                    (lambda x: (2 if x % 2 else 0), Operation.OR)]),
                                                   rand, scope, WeightsCollection([(10, 2)]),
                                                   WeightsCollection([(1, 2)])),
             lambda scope: RandomExpressionFactory(VariableWeights([(lambda x: (x - 1) * x, Operation.SYMBOL),
                                                                    (lambda x: (0 if x % 2 else 3), Operation.OR),
                                                                    (lambda x: (2 if x % 2 else 0), Operation.AND)]),
                                                   rand, scope, WeightsCollection([(10, 2)]),
                                                   WeightsCollection([(10, 2), (3, 3)])),
             lambda scope: RandomExpressionFactory(VariableWeights([(lambda x: (3) * x, Operation.SYMBOL),
                                                                    (lambda x: (0 if x % 2 else 3), Operation.OR),
                                                                    (lambda x: (3 if x % 2 else 0), Operation.AND)]),
                                                   rand, scope, WeightsCollection([(9, 2), (0.8, 3), (0.2, 4)]),
                                                   WeightsCollection([(1, 2), (7, 3), (2, 4)]))
             ]


def generateAnnotations(scope: Scope, fm: "FeatureModel", n: int, mn: float, mx: float, mean: Tuple[float, float]) -> \
        List[Expression]:
    fmWeight = len(fm.configurations)
    meanAvg = (mean[0] + mean[1]) / 2
    factory = factories[int(meanAvg * 5)](scope)
    filtered = 1
    annotations = []
    while True:
        expr = factory.newExpression(SymbolWeights(10, scope))
        avg = (expr & fm.expr).toCNF().getWeight()/fm.weight
        if mn < avg < mx:
            annotations.append(expr)
            break
    while len(annotations) < n:
        filtered += 1
        expr = factory.newExpression(SymbolWeights(10, scope))
        weight = (expr & fm.expr).toCNF().getWeight()/(fm.weight)
        if weight < mn or weight > mx:
            continue
        if mean[0] < avg < mean[1] and mean[0] < (avg * len(annotations) + weight) / (len(annotations) + 1) < mean[1]:
            annotations.append(expr)
            avg = (avg * len(annotations) + weight) / (len(annotations) + 1)
        elif mean[0] > avg and weight > avg:
            annotations.append(expr)
            avg = (avg * len(annotations) + weight) / (len(annotations) + 1)
        elif mean[1] < avg and weight < avg:
            annotations.append(expr)
            avg = (avg * len(annotations) + weight) / (len(annotations) + 1)
    print("Filtered {} annotations".format(filtered))
    return annotations


'''
This variable weight expression seems to generate some decently nice data.
vw = VariableWeights([(lambda x: 2*x, Operation.SYMBOL), (lambda x: i/(x+1), Operation.EQUALS),
                              (lambda x: 0/(x*x+1), Operation.IMPLIES), (lambda x: (2*i)/(x+1)*(1/5 if x%2 else 5), Operation.OR),
                              (lambda x: (8)/(x+1)*(5 if x%2 else 1/5), Operation.AND), (lambda x: 1/(x+1), Operation.NOT)])
i can be varied between 1 and 20 to get a variety of different weight peaks in order to generate several different distributions.
This can be used to generate the opposite sided list by negating the entire expression during generation (i.e swapping and and or to keep the same parity, and then making the NOT expression massive for x=0)
You can then apply demorgans law if you wish to turn this expression into a different looking one if you wish.
For some reason simply swapping and and or fails to generate the opposite sided distribution list.
'''
# TODO: Add some rules to make a tree simplify itself.
# This would involve defining a set of checks, for example True|x = True, ~x|x = True, ~x&x = False, etc...
# doing a leaf search first should make this only have to happen once.
if __name__ == '__main__':
    rand = random.Random()
    start = 0
    end = 1
    plt.hsv()
    scope = Scope(list(sympy.symbols(' '.join(string.ascii_uppercase[:8]))))
    fm = FeatureModel(factories[2](scope).newExpression(SymbolWeights(10, scope)))
    print(fm.weight)
    annotations = generateAnnotations(scope, fm, 20, 0.1, 0.5, (0.32, 0.34))
    for i in annotations:
        cnf = i.toCNF()
        print(i, cnf, (cnf&fm.expr.toCNF()).getWeight()/fm.weight)
