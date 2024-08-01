from typing import *

import pyapproxmc
import sympy
from bitarray import *
import random

from symbolic import Expression, Scope, RandomExpressionFactory, VariableWeights, Operation, SymbolWeights


# "configurations" here are just bitmask representations of the truth of variables. e.g. bit 0 could correspond to "A".

class TruthTable:
    def __init__(self, nVars: int, table):
        self.n = nVars
        if table is None:
            self.table = bitarray(1 << nVars)
        else:
            self.table = table
            assert len(table) == (1 << nVars)

    def populateTable(self, percentTruth: float):
        r = random
        for i in range(len(self.table)):
            if r.random() < percentTruth:  # populates each configuration with a percentTruth chance of being true
                self.table[i] = 1

    # or and and are basic operations that allow for a quick calculation of configurations with duplicates by taking the
    # disjunction of pairwise ands for identical values

    def __or__(self, other):
        assert other.n == self.n
        return TruthTable(self.n, other.table | self.table)

    def __and__(self, other):
        assert other.n == self.n
        return TruthTable(self.n, other.table & self.table)

    def __getitem__(self, item):
        return self.table[item]

    def __len__(self):
        return 1 << self.n

    def __repr__(self):
        return repr(self.table)


# really just a tuple with the ability to access the truth of the item given a configuration

class VariationalElement:
    def __init__(self, val: int, variation: Expression):
        self.val = val
        self.var = variation
        self.table = variation.getTruthTable()

    def __getitem__(self, ind):
        return self.table[ind]

    def __repr__(self):
        return "(" + str(self.val) + ", " + repr(self.var) + ")"

    def deg(self, fm: "FeatureModel"):
        if len(fm) == 0:
            return 0
        return sum(self.table[i] for i in fm.configurations) / len(fm)


class VariationalList:
    def __init__(self, lst: List[VariationalElement]):
        self.lst = lst

    def __getitem__(self, ind):
        return tuple([i.val for i in self.lst if i.table[ind]])

    def __repr__(self):
        return repr(self.lst)

    def __len__(self):
        return len(self.lst)

    def __iter__(self):
        return iter(self.lst)


class FeatureModel:
    def __init__(self, expr: Expression):
        self.table = expr.getTruthTable()
        self.expr = expr
        self.configurations = []
        for i in range(len(self.table)):
            if self.table[i] == 1:
                self.configurations.append(i)
        cnf = expr.toCNF()
        clauses = [i.toClause() for i in cnf.terms]
        mx = 0
        for i in clauses:
            mx = max(mx, max(map(abs, i)))
        c = pyapproxmc.Counter()
        c.add_clauses(clauses)
        count = c.count()
        self.weight = count[0]*2**(count[1]+len(expr.scope)-mx)


    def applyModelToList(self, lst: VariationalList):
        data = {}
        for configuration in self.configurations:
            product = lst[configuration]
            if product not in data:
                data[product] = []
            data[product].append(configuration)
        return ProductLine(self, data, lst)

    def __repr__(self):
        return repr(self.expr)

    def __len__(self):
        return len(self.configurations)


class ProductLine:
    def __init__(self, fm: FeatureModel, data: Dict[Tuple[int, ...], List[int]], lst: VariationalList):
        self.fm = fm
        self.data = data
        self.lst = lst

    def __len__(self):
        return len(self.data)

    def filterNoDuplicates(self):
        nodup = {}
        for i in self.data:
            if len(set(i)) == len(i):  # very naive way of finding duplicates but this runs in O(len*|fm|)
                nodup[i] = self.data[i]
        return ProductLine(self.fm, nodup, self.lst)

    @staticmethod
    def isSorted(lst):
        return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

    def filterSorted(self):
        sorted = {}
        for i in self.data:
            if ProductLine.isSorted(i):  # very naive way of finding duplicates but this runs in O(len*|fm|)
                sorted[i] = self.data[i]
        return ProductLine(self.fm, sorted, self.lst)

    def __repr__(self):
        return "Product Line over " + repr(self.lst) + " and data " + repr(self.data)

    def getAverageLength(self):
        if len(self.data) == 0:
            return 0
        return sum(map(len, self.data)) / len(self.data)

    def getAverageWeight(self):
        return sum(map(lambda x: x.deg(self.fm), self.lst)) / len(self.lst)

    def getWeights(self):
        return list(map(lambda x: x.deg(self.fm), self.lst))


def generateVariationalList(listSize: int, listDistribution: Callable[[], int], factory: RandomExpressionFactory):
    lst = []
    for i in range(listSize):
        # make the necessary calls to the random distributions, and generate a list.
        expr = factory.newExpression(SymbolWeights(5, factory.scope))
        lst.append(VariationalElement(listDistribution(), expr))
    return VariationalList(lst)


if __name__ == "__main__":
    scope = Scope(list(sympy.symbols("a b c d e")))
    vw = VariableWeights([(lambda x: 5 * x, Operation.SYMBOL), (lambda x: max(1, 2 - x), Operation.EQUALS),
                          (lambda x: max(1, 3 - x), Operation.IMPLIES), (lambda x: max(1, 5 - x), Operation.OR),
                          (lambda x: max(1, 4 - x), Operation.AND), (lambda x: 1, Operation.NOT)])
    rand = random.Random()
    factory = RandomExpressionFactory(vw, rand, scope)
    testCase = generateVariationalList(8, lambda: random.randint(1, 5), factory)
    print(testCase)
    symbolWeights = SymbolWeights(5, scope)
    fmTruth = factory.newExpression(symbolWeights)
    print(fmTruth)
    fm = FeatureModel(fmTruth)
    productLine = fm.applyModelToList(testCase)
    print(productLine)
    nodup = productLine.filterNoDuplicates()
    srted = productLine.filterSorted()
    print(nodup)
    print(srted)
    print(productLine.getAverageLength())
    print(nodup.getAverageLength())
    print(srted.getAverageLength())
    print(productLine.getAverageWeight())
    print(nodup.getAverageWeight())
    print(srted.getAverageWeight())
