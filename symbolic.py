import itertools
import random
from enum import Enum
from functools import reduce
from typing import Callable, List, Tuple, Dict
import argparse
import json

import pyapproxmc
import sympy
from z3 import *

class Scope:
    """
    A scope is used to define the variables used in generating problems.
    """
    def __init__(self, v: List[sympy.Symbol]):
        self.variables = v
        self.inverse = {v[i]: i for i in range(len(v))}

    def __len__(self):
        return len(self.variables)

    def __repr__(self):
        return "Symbolic Scope with the following variables:\n" + str(self.variables)


class Substitution:
    """
    A substitution maps variables to their truth values,
    these are used to evaluate variational types given configurations.
    """
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


class Expression:
    """
    This class represents any type of boolean expression that is used.
    It offers many useful methods such as conversion to CNF (used for model counting),
    conversion to z3 expressions for solving sharp-sat problems, and serialization.
    """
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

    @staticmethod
    def deserialize(ser: str, scope: Scope):
        globs = {}
        for var in scope.variables:
            globs[str(var)] = Expression(Operation.SYMBOL, [var], scope)
        # No other global types are offered so there shouldn't be any injection vulnerabilities
        return eval(ser, globs, {})
    def toZ3Expression(self):
        s = self.serialize()
        globs = {}
        for var in self.scope.variables:
            globs[str(var)] = Bool(str(var))
        return eval(s, globs, {})
    def serialize(self):
        if self.op == Operation.SYMBOL:
            return str(self.children[0])
        elif self.op == Operation.NOT:
            return "~" + self.children[0].serialize() if self.children[0].op == Operation.SYMBOL else "~(" + self.children[0].serialize() + ")"
        elif self.op == Operation.EQUALS:
            return "~(" + str(self.children[0]) + "^" + self.children[1].serialize() + ")"
        elif self.op == Operation.AND:
            return operationStrs[self.op.value].join(map(Expression.serialize, self.children))
        else:
            return "(" + operationStrs[self.op.value].join(map(Expression.serialize, self.children)) + ")"
    def __repr__(self):
        if self.op == Operation.SYMBOL:
            return str(self.children[0])
        elif self.op == Operation.NOT:
            return "~" + str(self.children[0]) if self.children[0].op == Operation.SYMBOL else "~(" + str(
                self.children[0]) + ")"
        elif self.op == Operation.EQUALS:
            return "~(" + str(self.children[0]) + "^" + str(self.children[1]) + ")"
        elif self.op == Operation.AND:
            return operationStrs[self.op.value].join(map(str, self.children))
        else:
            return "(" + operationStrs[self.op.value].join(map(str, self.children)) + ")"

    def applySubstitution(self, substitution: Substitution) -> bool:
        """
        Recursively applies a given substitution to an entire expression.
        :param substitution:
        :return: The truth value of this expression under the given configuration.
        """
        if self.op == Operation.SYMBOL:
            return substitution.sub[self.children[0]]
        return operationFuncs[self.op.value](*map(lambda x: x.applySubstitution(substitution), self.children))

    def __str__(self):
        return repr(self)

    def __or__(self, other: "Expression"):
        """
        Combines two expression by an OR operation.
        If either of self or other is an OR type operation, then the children are simplpy combined.
        :param other:
        :return:
        """
        assert self.scope == other.scope
        if self.op==Operation.OR and other.op == Operation.OR:
            return Expression(Operation.OR, self.children+other.children, self.scope)
        elif self.op==Operation.OR:
            return Expression(Operation.OR, self.children+[other], self.scope)
        elif other.op==Operation.OR:
            return Expression(Operation.OR, other.children+[self], self.scope)
        return Expression(Operation.OR, [self, other], self.scope)

    def __and__(self, other: "Expression"):
        """
        Combines two expression by an AND operation.
        If either of self or other is an AND type operation, then the children are simply combined.
        :param other:
        :return:
        """
        assert self.scope == other.scope
        if self.op==Operation.AND and other.op == Operation.AND:
            return Expression(Operation.AND, self.children+other.children, self.scope)
        elif self.op==Operation.AND:
            return Expression(Operation.AND, self.children+[other], self.scope)
        elif other.op==Operation.AND:
            return Expression(Operation.AND, other.children+[self], self.scope)
        return Expression(Operation.AND, [self, other], self.scope)

    def __invert__(self):
        return Expression(Operation.NOT, [self], self.scope)

    def __eq__(self, other: "Expression"):
        assert self.scope == other.scope
        return Expression(Operation.EQUALS, [self, other], self.scope)

    def toCNF(self):
        """
        Converts this expression to a CNF expression.
        This is done through the use of distributive properties and DeMorgan's law.
        :return:
        """
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


class FeatureModel:
    """
    This represents a feature model, which is used to describe a product line.
    Feature models offer sharp-sat solving using z3, in order to be able to list all configurations.
    """
    def __init__(self, expr: Expression):
        self.expr = expr
        self.cnf = expr.toCNF()
        clauses = [i.toClause() for i in self.cnf.terms]
        mx = 0
        for i in clauses:
            mx = max(mx, max(map(abs, i)))
        c = pyapproxmc.Counter()
        c.add_clauses(clauses)
        count = c.count()
        self.weight = count[0] * 2 ** (count[1] + len(expr.scope) - mx)
        self.configs = None
    @staticmethod
    def getBaseModels(s: z3.Solver):
        """
        Creates an iterable object of all models to the feature model expression.
        This is done by querying a z3 solver, and each time telling the solver that the
        last model is unsatisfiable, then querying it again until the overall problem is unsat.
        :param s:
        :return:
        """
        sats = s.check()
        while sats==sat:
            m = s.model()
            s.add(Or([v()!=m[v] for v in m]))
            sats = s.check()
            yield m
    @staticmethod
    def partialSubstitutionFiller(partialsub):
        """
        The solutions output by z3 don't always have assignments for all variables.
        This means that the other variables can be assigned to whatever.
        For example a scope of a, b, and c with feature model a|b has 6 solutions,
        but z3 will only give 2 (something like [a=True] and [a=False, b=True]).
        So this function generates all the complete configurations with the model that
        z3 outputs.
        :param partialsub:
        :return:
        """
        for i in range(len(partialsub)):
            if partialsub[i] is None:
                t = partialsub.copy()
                t[i] = True
                for s in FeatureModel.partialSubstitutionFiller(t):
                    yield s
                f = partialsub.copy()
                f[i] = False
                for s in FeatureModel.partialSubstitutionFiller(f):
                    yield s
                break
        else:
            #no unfilled gaps found
            yield partialsub
    def genConfigurations(self):
        """
        This generates all the configurations of the feature model so that they can be used
        to generate product lines.
        :return:
        """
        if self.configs is not None:
            return self.configs
        configs = []
        s = Solver()
        s.add(self.expr.toZ3Expression())
        for model in self.getBaseModels(s):
            partialsub = [None]*len(self.expr.scope)
            for v in model:
                partialsub[self.expr.scope.inverse[sympy.symbols(str(v))]] = bool(model[v])
            for s in self.partialSubstitutionFiller(partialsub):
                configs.append(Substitution({self.expr.scope.variables[i]:s[i] for i in range(len(s))}, self.expr.scope))
        self.configs = configs




    def __repr__(self):
        return repr(self.expr)

    def __len__(self):
        return self.weight


class Atom:
    """
    Describes a single variable and if it is negated.
    Used for creating CNF expressions.
    """
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
    """
    A single term of a CNF expression.
    This is the disjunction of several atoms.
    """
    def __init__(self, atoms: List[Atom], scope: Scope):
        self.atoms = atoms
        assert all(atoms[i].scope is scope for i in range(len(atoms)))
        self.scope = scope

    def __or__(self, other: "CNFTerm"):
        """
        The OR operation for two CNF terms.
        Combines their atoms and simplifies accordingly.
        :param other:
        :return:
        """
        assert self.scope is other.scope
        return CNFTerm(self.atoms + other.atoms, self.scope).simplify()

    def simplify(self):
        """
        Simplifies this CNF term.
        If an atom and its negation are both present, then the entire CNF term is simply true,
        denoted by a CNF term with no atoms.
        Any repeated atoms are also removed.
        :return:
        """
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
        """
        Defines whether this CNF term absorbs the other.
        For example the terms (a|b)&(a|b|c) Should simplify to a|b, since a and b are contained in a|b|c.
        :param other:
        :return:
        """
        return all(i in other.atoms for i in self.atoms)

    def __repr__(self):
        return '(' + '|'.join(map(str, self.atoms)) + ')'

    def __invert__(self):
        return CNF([CNFTerm([~i], self.scope) for i in self.atoms], self.scope)

    def __str__(self):
        return self.__repr__()

    def toClause(self):
        """
        Encodes this CNF term into a clause used for model counting in pyapproxmc.
        Assigns a positive or negative integer to each atom depending on the scopes
        encoding and its negation.
        :return:
        """
        return [(-(self.scope.inverse[i.symbol] + 1)) if i.negation else self.scope.inverse[i.symbol] + 1 for i in
                self.atoms]


class CNF:
    """
    A CNF expression.
    Used for model counting and generating full CNF expressions from regular expressions.
    """
    def __init__(self, terms: List[CNFTerm], scope: Scope):
        self.terms = terms
        assert all(terms[i].scope is scope for i in range(len(terms)))
        self.scope = scope

    def __and__(self, other: "CNF"):
        """
        Ands two CNF expression together by combining their children and simplifying.
        :param other:
        :return:
        """
        assert other.scope is self.scope
        return CNF(self.terms + other.terms, self.scope).simplify()

    def __or__(self, other: "CNF"):
        """
        Performs an OR operation by distributing and cross multiplying the CNF terms.
        :param other:
        :return:
        """
        assert other.scope is self.scope
        other.simplify()
        self.simplify()
        return CNF([i | j for i, j in itertools.product(self.terms, other.terms)], self.scope).simplify()

    def __invert__(self):
        """
        Uses DeMorgan's law to invert the expression.
        :return:
        """
        start = ~self.terms[0]
        for i in range(1, len(self.terms)):
            start |= ~self.terms[i]
        return start

    def simplify(self):
        """
        Simplifies the CNF expression.
        Any terms that need to be absorbed by another are absorbed.
        All terms are simplified themselves, and anything that evaluates to True is removed.
        :return:
        """
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
        """
        Invokes pyapproxmc to get the weight of this CNF expression.
        :return:
        """
        clauses = [i.toClause() for i in self.terms]
        mx = 0
        for i in clauses:
            mx = max(mx, max(map(abs, i)))
        c = pyapproxmc.Counter()
        c.add_clauses(clauses)
        count = c.count()
        weight = count[0] * 2 ** (count[1] + len(self.scope) - mx)
        return weight


class WeightsCollection:
    """
    A list of weighted objects to randomly select from.
    """
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
    """
    Defines a function which determines the weight of various objects based on a parameter.
    Used for changing the probabilities of each type of expression being continued depending on the depth in the tree.
    """
    def __init__(self, weights: List[Tuple[Callable[[int], float], Operation]]):
        self.cache = {}
        self.weights = weights

    def __getitem__(self, index) -> WeightsCollection:
        if index not in self.cache:
            weight = [(self.weights[i][0](index), self.weights[i][1]) for i in range(len(self.weights))]
            self.cache[index] = WeightsCollection(weight)
        return self.cache[index]


class SymbolWeights:
    """
    Gives an interface for getting random symbols for expression generation.
    Once a variable is given, its probability is divided by a factor to prevent frequent repeated symbols.
    """
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
    """
    This factory generates random expressions given variable weights and several other configurations.
    """
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
        # Depending on operation generate any additional nodes recursively.
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
            return Expression(op, [self.newExpression(sw, depth + 1) for _ in range(self.andWeights[self.rand])],
                              self.scope)
        elif op == Operation.OR:
            return Expression(op, [self.newExpression(sw, depth + 1) for _ in range(self.orWeights[self.rand])],
                              self.scope)

depthFactor = 1.3
factories = [lambda scope: RandomExpressionFactory(VariableWeights([(lambda x: depthFactor * 3 * x, Operation.SYMBOL),
                                                                    (lambda x: (0 if x % 2 else 3), Operation.AND),
                                                                    (lambda x: (3 if x % 2 else 0), Operation.OR)]),
                                                   random, scope, WeightsCollection([(1, 2), (7, 3), (3, 4)]),
                                                   WeightsCollection([(9, 2), (0.8, 3), (0.2, 4)])),
             lambda scope: RandomExpressionFactory(VariableWeights([(lambda x: depthFactor * (x - .5) * x, Operation.SYMBOL),
                                                                    (lambda x: (0 if x % 2 else 3), Operation.AND),
                                                                    (lambda x: (2 if x % 2 else 0), Operation.OR)]),
                                                   random, scope, WeightsCollection([(10, 2), (3, 3)]),
                                                   WeightsCollection([(10, 2)])),
             lambda scope: RandomExpressionFactory(VariableWeights([(lambda x: depthFactor * 1.5 * (x - .5) * x, Operation.SYMBOL),
                                                                    (lambda x: (0 if x % 2 else 2), Operation.AND),
                                                                    (lambda x: (2 if x % 2 else 0), Operation.OR)]),
                                                   random, scope, WeightsCollection([(10, 2)]),
                                                   WeightsCollection([(1, 2)])),
             lambda scope: RandomExpressionFactory(VariableWeights([(lambda x: depthFactor * (x - .5) * x, Operation.SYMBOL),
                                                                    (lambda x: (0 if x % 2 else 3), Operation.OR),
                                                                    (lambda x: (2 if x % 2 else 0), Operation.AND)]),
                                                   random, scope, WeightsCollection([(10, 2)]),
                                                   WeightsCollection([(10, 2), (3, 3)])),
             lambda scope: RandomExpressionFactory(VariableWeights([(lambda x: depthFactor * 3 * x, Operation.SYMBOL),
                                                                    (lambda x: (0 if x % 2 else 3), Operation.OR),
                                                                    (lambda x: (3 if x % 2 else 0), Operation.AND)]),
                                                   random, scope, WeightsCollection([(9, 2), (0.8, 3), (0.2, 4)]),
                                                   WeightsCollection([(1, 2), (7, 3), (2, 4)]))
             ]


def generateAnnotations(scope: Scope, fm: "FeatureModel", n: int, mn: float, mx: float, mean: Tuple[float, float]) -> \
        List[Expression]:
    meanAvg = (mean[0] + mean[1]) / 2
    factory = factories[int(meanAvg * 5)](scope)
    filtered = 1
    annotations = []
    while True:
        expr = factory.newExpression(SymbolWeights(10, scope))
        avg = (expr & fm.expr).toCNF().getWeight() / fm.weight
        if mn < avg < mx:
            annotations.append(expr)
            break
    while len(annotations) < n:
        filtered += 1
        expr = factory.newExpression(SymbolWeights(10, scope))
        weight = (expr & fm.expr).toCNF().getWeight() / fm.weight
        if weight < mn or weight > mx:
            continue
        if mean[0] <= avg <= mean[1] and mean[0] <= (avg * len(annotations) + weight) / (len(annotations) + 1) <= mean[1]:
            annotations.append(expr)
            avg = (avg * len(annotations) + weight) / (len(annotations) + 1)
        elif mean[0] > avg and weight > avg:
            annotations.append(expr)
            avg = (avg * len(annotations) + weight) / (len(annotations) + 1)
        elif mean[1] < avg and weight < avg:
            annotations.append(expr)
            avg = (avg * len(annotations) + weight) / (len(annotations) + 1)
        print(len(annotations), weight, avg, (avg * len(annotations) + weight) / (len(annotations) + 1))
    print("Filtered {} annotations".format(filtered))
    return annotations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate annotations under a given feature model')
    parser.add_argument('command', choices=['annotations', 'featuremodel'])
    parser.add_argument('input', type=argparse.FileType('r'))
    parser.add_argument('output', type=str)
    args = parser.parse_args()
    command = args.command

    config = json.loads(args.input.read())
    out = open(args.output, 'w')
    if command == 'annotations':
        for var in config['vars']:
            if any(i in var for i in ' &|~(),'):
                print("Invalid variable name: " + var)
                exit(1)
        scope = Scope(list(sympy.symbols(' '.join(config['vars']))))
        fm = FeatureModel(Expression.deserialize(config['fm'], scope))
        print("Loaded feature model and scope")
        annotations = generateAnnotations(scope, fm, config['n'], config['min'], config['max'], config['mean'])
        print("Generated")
        config["annotations"] = list(map(lambda x: x.serialize(), annotations))
        out.write(json.dumps(config))
        out.close()
    elif command == 'featuremodel':
        for var in config['vars']:
            if any(i in var for i in ' &|~()'):
                print("Invalid variable name: " + var)
                exit(1)
        scope = Scope(list(sympy.symbols(' '.join(config['vars']))))
        fm = FeatureModel(factories[2](scope).newExpression(SymbolWeights(10, scope)))
        config['fm'] = fm.expr.serialize()
        out.write(json.dumps(config))
        out.close()
