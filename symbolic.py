import random
from enum import Enum
from typing import Callable, List, Tuple, Dict

import sympy
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
    IMPLIES = 4
    EQUALS = 5


operationStrs = [None, "~", "&", "|", "=>", "=="]
operationFuncs = [None, lambda x: not x, lambda x, y: x & y, lambda x, y: x | y, lambda x, y: (not x) | y,
                  lambda x, y: x == y]
truthTableFuncs = [None, lambda x: ~x, lambda x, y: x & y, lambda x, y: x | y, lambda x, y: ~x | y,
                   lambda x, y: ~(x ^ y)]


class Expression:
    def __init__(self, op, children: List[sympy.Symbol or "Expression"], scope: Scope):
        if op == Operation.SYMBOL:
            assert len(children) == 1
            assert type(children[0]) is sympy.Symbol
        elif op == Operation.NOT:
            assert len(children) == 1
        else:
            assert len(children) == 2
        self.op = op
        self.children = children
        self.scope = scope

    def __repr__(self):
        if self.op == Operation.SYMBOL:
            return str(self.children[0])
        elif self.op == Operation.NOT:
            return "(~" + str(self.children[0]) + ")"
        else:
            return "(" + str(self.children[0]) + operationStrs[self.op.value] + str(self.children[1]) + ")"

    def applySubstitution(self, substitution: Substitution):
        if self.op == Operation.SYMBOL:
            return substitution.sub[self.children[0]]
        return operationFuncs[self.op.value](*map(lambda x: x.applySubstitution(substitution), self.children))
        # kinda unsafe in terms of arguments, but it should
        # be fine because of the assertions in __init__

    def getTruthTable(self):
        if self.op == Operation.SYMBOL:
            return self.scope.truthTables[self.scope.inverse[self.children[0]]]
        return truthTableFuncs[self.op.value](*map(lambda x: x.getTruthTable(), self.children))

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

    def __ge__(self, other: "Expression"):
        assert self.scope == other.scope
        return Expression(Operation.IMPLIES, [self, other], self.scope)


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


class RandomExpressionFactory:
    def __init__(self, weights: VariableWeights, rand: random.Random, scope: Scope):
        self.weights = weights
        self.rand = rand
        self.scope = scope

    def newExpression(self, depth=0) -> Expression:
        weight = self.weights[depth]
        op = weight[self.rand]
        if op == Operation.SYMBOL:
            return Expression(op, [self.rand.choice(self.scope.variables)], self.scope)
        elif op == Operation.NOT:
            return Expression(op, [self.newExpression(depth + 1)], self.scope)
        else:
            return Expression(op, [self.newExpression(depth + 1), self.newExpression(depth + 1)], self.scope)


if __name__ == '__main__':
    rand = random.Random()
    scope = Scope(list(sympy.symbols("a b c d e")))
    print(scope)
    vw = VariableWeights([(lambda x: 2 * x, Operation.SYMBOL), (lambda x: max(1, 5 - x), Operation.EQUALS),
                          (lambda x: max(1, 5 - x), Operation.IMPLIES), (lambda x: max(1, 5 - x), Operation.OR),
                          (lambda x: max(1, 5 - x), Operation.AND), (lambda x: 1, Operation.NOT)])
    factory = RandomExpressionFactory(vw, rand, scope)
    expr = factory.newExpression()
    print(expr)
    sub = Substitution(dict([(i, bool(random.getrandbits(1))) for i in scope.variables]), scope)
    print(sub)
    print(expr.applySubstitution(sub))
    print(expr.getTruthTable())
