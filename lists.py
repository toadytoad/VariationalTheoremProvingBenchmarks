from typing import *
import json

import sympy
from bitarray import *
import random

from symbolic import Expression, Scope, RandomExpressionFactory, VariableWeights, Operation, SymbolWeights, FeatureModel


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


    def __repr__(self):
        return "(" + str(self.val) + ", " + repr(self.var) + ")"



class VariationalList:
    def __init__(self, lst: List[VariationalElement]):
        self.lst = lst

    def __getitem__(self, ind):
        return self.lst[ind]

    def __repr__(self):
        return repr(self.lst)

    def __len__(self):
        return len(self.lst)

    def __iter__(self):
        return iter(self.lst)





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

# Length of list
# Choose elements from 0 to n-1
# x contiguous sublists of size at least k with the same annotations
# at least ... elements appear m_i times.


def generateVariationalList(listSize: int, elements:Set,elementRepetitions:List[int], contiguousSublists:List[int], annotationBank:List[Expression]):
    lst = []
    for i in elementRepetitions:
        element = random.choice(list(elements))
        elements.remove(element)
        lst+=[element]*i
    assert len(lst) <= listSize
    lst+=random.choices(list(elements), k=listSize-len(lst))
    random.shuffle(lst)
    assert len(contiguousSublists) <= listSize-sum(contiguousSublists)
    indices = sorted(random.sample(range(listSize-sum(contiguousSublists)+len(contiguousSublists)), k=2*len(contiguousSublists)))
    annotations = [None]*listSize
    print(indices)
    x = 0
    random.shuffle(contiguousSublists)
    for i in range(0, len(indices), 2):
        start = indices[i]+x
        x+=contiguousSublists[i//2]-1
        end = indices[i+1]+x
        print(start, end)
        annotation = random.choice(annotationBank)
        annotationBank.remove(annotation)
        for j in range(start, end+1):
            annotations[j] = annotation
    for i in range(listSize):
        if annotations[i] is None:
            annotations[i] = random.choice(annotationBank)
            annotationBank.remove(annotations[i])

    return [VariationalElement(i, j) for i, j in zip(lst, annotations)]


if __name__ == "__main__":
    config = json.loads(open("annotations.json").read())
    scope = Scope(list(sympy.symbols(' '.join(config['vars']))))
    annotations = list(map(lambda x: Expression.deserialize(x, scope), config["annotations"]))
    print(*generateVariationalList(20, set(range(20)), [2, 2, 3], [3, 2, 4], annotations), sep='\n')
